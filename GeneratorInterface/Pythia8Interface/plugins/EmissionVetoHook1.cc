#include "Pythia8/Pythia.h"

using namespace Pythia8;

#include "GeneratorInterface/Pythia8Interface/plugins/EmissionVetoHook1.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
void EmissionVetoHook1::fatalEmissionVeto(std::string message) {
  throw edm::Exception(edm::errors::Configuration,"Pythia8Interface")
    << "EmissionVeto: " << message << std::endl;
}

//--------------------------------------------------------------------------

// Routines to calculate the pT (according to pTdefMode) in a splitting:
//   ISR: i (radiator after)  -> j (emitted after) k (radiator before)
//   FSR: i (radiator before) -> j (emitted after) k (radiator after)
// For the Pythia pT definition, a recoiler (after) must be specified.

// Compute the Pythia pT separation. Based on pTLund function in History.cc
double EmissionVetoHook1::pTpythia(const Pythia8::Event &e, int RadAfterBranch,
                                   int EmtAfterBranch, int RecAfterBranch, bool FSR) {

  // Convenient shorthands for later
  Pythia8::Vec4 radVec = e[RadAfterBranch].p();
  Pythia8::Vec4 emtVec = e[EmtAfterBranch].p();
  Pythia8::Vec4 recVec = e[RecAfterBranch].p();
  int  radID  = e[RadAfterBranch].id();

  // Calculate virtuality of splitting
  double sign = (FSR) ? 1. : -1.;
  Pythia8::Vec4 Q(radVec + sign * emtVec); 
  double Qsq = sign * Q.m2Calc();

  // Mass term of radiator
  double m2Rad = (abs(radID) >= 4 && abs(radID) < 7) ?
                  Pythia8::pow2(particleDataPtr->m0(radID)) : 0.;

  // z values for FSR and ISR
  double z, pTnow;
  if (FSR) {
    // Construct 2 -> 3 variables
    Pythia8::Vec4 sum = radVec + recVec + emtVec;
    double m2Dip = sum.m2Calc();
    double x1 = 2. * (sum * radVec) / m2Dip;
    double x3 = 2. * (sum * emtVec) / m2Dip;
    z     = x1 / (x1 + x3);
    pTnow = z * (1. - z);

  } else {
    // Construct dipoles before/after splitting
    Pythia8::Vec4 qBR(radVec - emtVec + recVec);
    Pythia8::Vec4 qAR(radVec + recVec);
    z     = qBR.m2Calc() / qAR.m2Calc();
    pTnow = (1. - z);
  }

  // Virtuality with correct sign
  pTnow *= (Qsq - sign * m2Rad);

  // Can get negative pT for massive splittings
  if (pTnow < 0.) {
    std::cout << "Warning: pTpythia was negative" << std::endl;
    return -1.;
  }

#ifdef DBGOUTPUT
  std::cout << "pTpythia: rad = " << RadAfterBranch << ", emt = "
       << EmtAfterBranch << ", rec = " << RecAfterBranch
       << ", pTnow = " << sqrt(pTnow) << std::endl;
#endif

  // Return pT
  return sqrt(pTnow);
}

// Compute the POWHEG pT separation between i and j
double EmissionVetoHook1::pTpowheg(const Pythia8::Event &e, int i, int j, bool FSR) {

  // pT value for FSR and ISR
  double pTnow = 0.;
  if (FSR) {
    // POWHEG d_ij (in CM frame). Note that the incoming beams have not
    // been updated in the parton systems pointer yet (i.e. prior to any
    // potential recoil).
    int iInA = partonSystemsPtr->getInA(0);
    int iInB = partonSystemsPtr->getInB(0);
    double betaZ = - ( e[iInA].pz() + e[iInB].pz() ) /
                     ( e[iInA].e()  + e[iInB].e()  );
    Pythia8::Vec4 iVecBst(e[i].p()), jVecBst(e[j].p());
    iVecBst.bst(0., 0., betaZ);
    jVecBst.bst(0., 0., betaZ);
    pTnow = sqrt( (iVecBst + jVecBst).m2Calc() *
                  iVecBst.e() * jVecBst.e() /
                  Pythia8::pow2(iVecBst.e() + jVecBst.e()) );

  } else {
    // POWHEG pT_ISR is just kinematic pT
    pTnow = e[j].pT();
  }

  // Check result
  if (pTnow < 0.) {
    std::cout << "Warning: pTpowheg was negative" << std::endl;
    return -1.;
  }

#ifdef DBGOUTPUT
  std::cout << "pTpowheg: i = " << i << ", j = " << j
       << ", pTnow = " << pTnow << std::endl;
#endif

   return pTnow;
}

// Calculate pT for a splitting based on pTdefMode.
// If j is -1, all final-state partons are tried.
// If i, k, r and xSR are -1, then all incoming and outgoing 
// partons are tried.
// xSR set to 0 means ISR, while xSR set to 1 means FSR
double EmissionVetoHook1::pTcalc(const Pythia8::Event &e, int i, int j, int k, int r, int xSRin) {

  // Loop over ISR and FSR if necessary
  double pTemt = -1., pTnow;
  int xSR1 = (xSRin == -1) ? 0 : xSRin;
  int xSR2 = (xSRin == -1) ? 2 : xSRin + 1;
  for (int xSR = xSR1; xSR < xSR2; xSR++) {
    // FSR flag
    bool FSR = (xSR == 0) ? false : true;

    // If all necessary arguments have been given, then directly calculate.
    // POWHEG ISR and FSR, need i and j.
    if ((pTdefMode == 0 || pTdefMode == 1) && i > 0 && j > 0) {
      pTemt = pTpowheg(e, i, j, (pTdefMode == 0) ? false : FSR);

    // Pythia ISR, need i, j and r.
    } else if (!FSR && pTdefMode == 2 && i > 0 && j > 0 && r > 0) {
      pTemt = pTpythia(e, i, j, r, FSR);

    // Pythia FSR, need k, j and r.
    } else if (FSR && pTdefMode == 2 && j > 0 && k > 0 && r > 0) {
      pTemt = pTpythia(e, k, j, r, FSR);

    // Otherwise need to try all possible combintations.
    } else {
      // Start by finding incoming legs to the hard system after
      // branching (radiator after branching, i for ISR).
      // Use partonSystemsPtr to find incoming just prior to the
      // branching and track mothers.
      int iInA = partonSystemsPtr->getInA(0);
      int iInB = partonSystemsPtr->getInB(0);
      while (e[iInA].mother1() != 1) { iInA = e[iInA].mother1(); }
      while (e[iInB].mother1() != 2) { iInB = e[iInB].mother1(); }

      // If we do not have j, then try all final-state partons
      int jNow = (j > 0) ? j : 0;
      int jMax = (j > 0) ? j + 1 : e.size();
      for (; jNow < jMax; jNow++) {

        // Final-state only 
        if (!e[jNow].isFinal()) continue;
	// Exclude photons (and W/Z!)
	if (QEDvetoMode==0 && e[jNow].colType() == 0) continue;

        // POWHEG
        if (pTdefMode == 0 || pTdefMode == 1) {

          // ISR - only done once as just kinematical pT
          if (!FSR) {
            pTnow = pTpowheg(e, iInA, jNow, (pTdefMode == 0) ? false : FSR);
            if (pTnow > 0.) pTemt = (pTemt < 0) ? pTnow : std::min(pTemt, pTnow);
  
          // FSR - try all outgoing partons from system before branching 
          // as i. Note that for the hard system, there is no 
          // "before branching" information.
          } else {
    
            int outSize = partonSystemsPtr->sizeOut(0);
            for (int iMem = 0; iMem < outSize; iMem++) {
              int iNow = partonSystemsPtr->getOut(0, iMem);

              // if i != jNow and no carbon copies
              if (iNow == jNow) continue;
	      // Exlude photons (and W/Z!) 
	      if (QEDvetoMode==0 && e[iNow].colType() == 0) continue;
              if (jNow == e[iNow].daughter1() 
                  && jNow == e[iNow].daughter2()) continue;

              pTnow = pTpowheg(e, iNow, jNow, (pTdefMode == 0) 
                ? false : FSR);
              if (pTnow > 0.) pTemt = (pTemt < 0) 
                ? pTnow : std::min(pTemt, pTnow);
            } // for (iMem)
  
          } // if (!FSR)
  
        // Pythia
        } else if (pTdefMode == 2) {
  
          // ISR - other incoming as recoiler
          if (!FSR) {
            pTnow = pTpythia(e, iInA, jNow, iInB, FSR);
            if (pTnow > 0.) pTemt = (pTemt < 0) ? pTnow : std::min(pTemt, pTnow);
            pTnow = pTpythia(e, iInB, jNow, iInA, FSR);
            if (pTnow > 0.) pTemt = (pTemt < 0) ? pTnow : std::min(pTemt, pTnow);
  
          // FSR - try all final-state coloured partons as radiator
          //       after emission (k).
          } else {
	    for (int kNow = 0; kNow < e.size(); kNow++) {
	      if (kNow == jNow || !e[kNow].isFinal()) continue;
              if (QEDvetoMode==0 && e[kNow].colType() == 0) continue;
  
              // For this kNow, need to have a recoiler.
              // Try two incoming.
              pTnow = pTpythia(e, kNow, jNow, iInA, FSR);
              if (pTnow > 0.) pTemt = (pTemt < 0) 
                ? pTnow : std::min(pTemt, pTnow);
              pTnow = pTpythia(e, kNow, jNow, iInB, FSR);
              if (pTnow > 0.) pTemt = (pTemt < 0) 
                ? pTnow : std::min(pTemt, pTnow);

              // Try all other outgoing.
              for (int rNow = 0; rNow < e.size(); rNow++) {
                if (rNow == kNow || rNow == jNow ||
                    !e[rNow].isFinal()) continue;
		if(QEDvetoMode==0 && e[rNow].colType() == 0) continue;
                pTnow = pTpythia(e, kNow, jNow, rNow, FSR);
                if (pTnow > 0.) pTemt = (pTemt < 0) 
                  ? pTnow : std::min(pTemt, pTnow);
              } // for (rNow)
  
            } // for (kNow)
          } // if (!FSR)
        } // if (pTdefMode)
      } // for (j)
    }
  } // for (xSR)

#ifdef DBGOUTPUT
  std::cout << "pTcalc: i = " << i << ", j = " << j << ", k = " << k
       << ", r = " << r << ", xSR = " << xSRin
       << ", pTemt = " << pTemt << std::endl;
#endif

  return pTemt;
}

//--------------------------------------------------------------------------

// Extraction of pThard based on the incoming event.
// Assume that all the final-state particles are in a continuous block
// at the end of the event and the final entry is the POWHEG emission.
// If there is no POWHEG emission, then pThard is set to QRen.
bool EmissionVetoHook1::doVetoMPIStep(int nMPI, const Pythia8::Event &e) {

    if(nFinalMode == 3 && pThardMode != 0) 
      fatalEmissionVeto(std::string("When nFinalMode is set to 3, ptHardMode should be set to 0, since the emission variables in doVetoMPIStep are not set correctly case when there are three possible particle Born particle counts."));

  // Extra check on nMPI
  if (nMPI > 1) return false;

  // Find if there is a POWHEG emission. Go backwards through the
  // event record until there is a non-final particle. Also sum pT and
  // find pT_1 for possible MPI vetoing
  // Flag if POWHEG radiation is present and index at the same time
  int count = 0, inonfinal = 0;
  double pT1 = 0., pTsum = 0.;
  isEmt = false;
  int iEmt = -1; 
  
  for (int i = e.size() - 1; i > 0; i--) {
    inonfinal = i;
    if (e[i].isFinal()) {
      count++;
      pT1    = e[i].pT();
      pTsum += e[i].pT();
      // the following was added for bbbar4l and will be triggered by specifying nfinalmode == 2 
      // the solution provided by Tomas may not be process independent but should work for hvq and bb4l
      // if the particle is a light quark or a gluon and 
      // comes for a light quark or a gluon (not a resonance, not a top)
      // then it is the POWHEG emission (hvq) or the POWHEG emission in production (bb4l)
      if (nFinalMode == 2) {
	if ((abs(e[i].id()) < 6 || e[i].id() == 21) && 
	    (abs(e[e[i].mother1()].id()) < 6 || e[e[i].mother1()].id() == 21)) {
	  isEmt = true;
	  iEmt = i;
	}
      } 
    } else break;
  }
  
  nFinal = nFinalExt;
  if (nFinal < 0 || nFinalMode == 1) {      // nFinal is not specified from external, try to find out
    int first = -1, myid;
    int last = -1;
    for(int ip = 2; ip < e.size(); ip++) {
      myid = e[ip].id();
      if(abs(myid) < 6 || abs(myid) == 21) continue;
      first = ip;
      break;
    }
    if(first < 0) fatalEmissionVeto(std::string("signal particles not found"));
    for(int ip = first; ip < e.size(); ip++) {
      myid = e[ip].id();
      if(abs(myid) < 6 || abs(myid) == 21) continue;
      last = ip;
    }
    nFinal = last - inonfinal;
  }

  // don't perform a cross check in case of nfinalmode == 2 
  if (nFinalMode != 2) { 

    // Extra check that we have the correct final state
    // In POWHEG WGamma, both w+0/1 jets and w+gamma+0/1 jets are generated at the same time, which leads to three different possible numbers of particles
    // Normally, there would be only two possible numbers of particles for X and X+1 jet
    if(nFinalMode == 3){
      if (count != nFinal && count != nFinal + 1 && count != nFinal - 1)
	fatalEmissionVeto(std::string("Wrong number of final state particles in event"));
    } else {
      if (count != nFinal && count != nFinal + 1)
	fatalEmissionVeto(std::string("Wrong number of final state particles in event"));
    }
    // Flag if POWHEG radiation present and index
    if (count == nFinal + 1) isEmt = true;
    if (isEmt) iEmt = e.size() - 1;
  } 
  
  // If there is no radiation or if pThardMode is 0 then set pThard to QRen.
  if (!isEmt || pThardMode == 0) {
    pThard = infoPtr->QRen();
      
  // If pThardMode is 1 then the pT of the POWHEG emission is checked against
  // all other incoming and outgoing partons, with the minimal value taken
  } else if (pThardMode == 1) {
    pThard = pTcalc(e, -1, iEmt, -1, -1, -1);

  // If pThardMode is 2, then the pT of all final-state partons is checked
  // against all other incoming and outgoing partons, with the minimal value
  // taken
  } else if (pThardMode == 2) {
    pThard = pTcalc(e, -1, -1, -1, -1, -1);

  }

  // Find MPI veto pT if necessary
  if (MPIvetoOn) {
    pTMPI = (isEmt) ? pTsum / 2. : pT1;
  }

  if(Verbosity)
    std::cout << "doVetoMPIStep: QFac = " << infoPtr->QFac()
         << ", QRen = " << infoPtr->QRen()
         << ", pThard = " << pThard << std::endl << std::endl;

  // Initialise other variables
  accepted   = false;
  nAcceptSeq = 0; // should not  reset nISRveto = nFSRveto = nFSRvetoBB4l here 

  // Do not veto the event
  return false;
}

//--------------------------------------------------------------------------

// ISR veto

bool EmissionVetoHook1::doVetoISREmission(int, const Pythia8::Event &e, int iSys) {
  // Must be radiation from the hard system
  if (iSys != 0) return false;

  // If we already have accepted 'vetoCount' emissions in a row, do nothing
  if (vetoOn && nAcceptSeq >= vetoCount) return false;

  // Pythia radiator after, emitted and recoiler after.
  int iRadAft = -1, iEmt = -1, iRecAft = -1;
  for (int i = e.size() - 1; i > 0; i--) {
    if      (iRadAft == -1 && e[i].status() == -41) iRadAft = i;
    else if (iEmt    == -1 && e[i].status() ==  43) iEmt    = i;
    else if (iRecAft == -1 && e[i].status() == -42) iRecAft = i;
    if (iRadAft != -1 && iEmt != -1 && iRecAft != -1) break;
  }
  if (iRadAft == -1 || iEmt == -1 || iRecAft == -1) {
    e.list();
    fatalEmissionVeto(std::string("Couldn't find Pythia ISR emission"));
  }

  // pTemtMode == 0: pT of emitted w.r.t. radiator
  // pTemtMode == 1: std::min(pT of emitted w.r.t. all incoming/outgoing)
  // pTemtMode == 2: std::min(pT of all outgoing w.r.t. all incoming/outgoing)
  int xSR      = (pTemtMode == 0) ? 0       : -1;
  int i        = (pTemtMode == 0) ? iRadAft : -1;
  int j        = (pTemtMode != 2) ? iEmt    : -1;
  int k        = -1;
  int r        = (pTemtMode == 0) ? iRecAft : -1;
  double pTemt = pTcalc(e, i, j, k, r, xSR);

#ifdef DBGOUTPUT
  std::cout << "doVetoISREmission: pTemt = " << pTemt << std::endl << std::endl;
#endif

  // If a Born configuration, and a colorless FS, and QEDvetoMode=2,
  // then don't veto photons, W, or Z harder than pThard
  bool vetoParton = (!isEmt && e[iEmt].colType()==0 && QEDvetoMode==2)
      ? false: true;
  
  // Veto if pTemt > pThard
  if (pTemt > pThard) {
    if(!vetoParton) {
      // Don't veto ANY emissions afterwards 
      nAcceptSeq = vetoCount-1;
    } else {	  
      nAcceptSeq = 0;
      nISRveto++;
    return true;
    }
  }

  // Else mark that an emission has been accepted and continue
  nAcceptSeq++;
  accepted = true;
  return false;
}

//--------------------------------------------------------------------------

// FSR veto

bool EmissionVetoHook1::doVetoFSREmission(int, const Pythia8::Event &e, int iSys, bool inResonance) {

  // Must be radiation from the hard system
  if (iSys != 0) return false;

  // only use for outside resonance vetos in combination with bb4l:FSREmission:veto
  if (inResonance && settingsPtr->flag("POWHEG:bb4l:FSREmission:veto")==1) return false;

  // If we already have accepted 'vetoCount' emissions in a row, do nothing
  if (vetoOn && nAcceptSeq >= vetoCount) return false;

  // Pythia radiator (before and after), emitted and recoiler (after)
  int iRecAft = e.size() - 1;
  int iEmt    = e.size() - 2;
  int iRadAft = e.size() - 3;
  int iRadBef = e[iEmt].mother1();
  if ( (e[iRecAft].status() != 52 && e[iRecAft].status() != -53) ||
       e[iEmt].status() != 51 || e[iRadAft].status() != 51) {
    e.list();
    fatalEmissionVeto(std::string("Couldn't find Pythia FSR emission"));
  }

  // Behaviour based on pTemtMode:
  //  0 - pT of emitted w.r.t. radiator before
  //  1 - std::min(pT of emitted w.r.t. all incoming/outgoing)
  //  2 - std::min(pT of all outgoing w.r.t. all incoming/outgoing)
  int xSR = (pTemtMode == 0) ? 1       : -1;
  int i   = (pTemtMode == 0) ? iRadBef : -1;
  int k   = (pTemtMode == 0) ? iRadAft : -1;
  int r   = (pTemtMode == 0) ? iRecAft : -1;

  // When pTemtMode is 0 or 1, iEmt has been selected
  double pTemt = -1.;
  if (pTemtMode == 0 || pTemtMode == 1) {
    // Which parton is emitted, based on emittedMode:
    //  0 - Pythia definition of emitted
    //  1 - Pythia definition of radiated after emission
    //  2 - Random selection of emitted or radiated after emission
    //  3 - Try both emitted and radiated after emission
    int j = iRadAft;
    if (emittedMode == 0 || (emittedMode == 2 && rndmPtr->flat() < 0.5)) j++;

    for (int jLoop = 0; jLoop < 2; jLoop++) {
      if      (jLoop == 0) pTemt = pTcalc(e, i, j, k, r, xSR);
      else if (jLoop == 1) pTemt = std::min(pTemt, pTcalc(e, i, j, k, r, xSR));
  
      // For emittedMode == 3, have tried iRadAft, now try iEmt
      if (emittedMode != 3) break;
      if (k != -1) std::swap(j, k); else j = iEmt;
    }

  // If pTemtMode is 2, then try all final-state partons as emitted
  } else if (pTemtMode == 2) {
    pTemt = pTcalc(e, i, -1, k, r, xSR);

  }

#ifdef DBGOUTPUT
  std::cout << "doVetoFSREmission: pTemt = " << pTemt << std::endl << std::endl;
#endif

  // If a Born configuration, and a colorless FS, and QEDvetoMode=2,
  // then don't veto photons, W, or Z harder than pThard
  bool vetoParton = (!isEmt && e[iEmt].colType()==0 && QEDvetoMode==2)
      ? false: true;
  
  // Veto if pTemt > pThard
  if (pTemt > pThard) {
    if(!vetoParton) {
      // Don't veto ANY emissions afterwards 
      nAcceptSeq = vetoCount-1;
    } else {	    
      nAcceptSeq = 0;
      nFSRveto++;
      return true;
    }
  }
  
  // Else mark that an emission has been accepted and continue
  nAcceptSeq++;
  accepted = true;
  return false;
}

//--------------------------------------------------------------------------

// MPI veto

bool EmissionVetoHook1::doVetoMPIEmission(int, const Pythia8::Event &e) {
  if (MPIvetoOn) {
    if (e[e.size() - 1].pT() > pTMPI) {
#ifdef DBGOUTPUT
      std::cout << "doVetoMPIEmission: pTnow = " << e[e.size() - 1].pT()
		<< ", pTMPI = " << pTMPI << std::endl << std::endl;
#endif
      return true;
    }
  }
  return false;
}

