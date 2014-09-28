#include "GeneratorInterface/Pythia8Interface/plugins/EmissionVetoHook.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
void EmissionVetoHook::fatalEmissionVeto(string message) {
  throw edm::Exception(edm::errors::Configuration,"Pythia8Interface")
    << "EmissionVeto: " << message << endl;
}

// Use VetoMIStep to analyse the incoming LHEF event and
// extract the veto scale
bool EmissionVetoHook::doVetoMPIStep(int, const Pythia8::Event &e) {
  int first=-1, myid;
  last = -1;
  for(int ip = 2; ip < e.size(); ip++) {
    myid = e[ip].id();
    if(abs(myid) < 6 || abs(myid) == 21) continue;
    first = ip;
    break;
  }
  if(first < 0) fatalEmissionVeto(string("signal particles not found"));
  for(int ip = first; ip < e.size(); ip++) {
    myid = e[ip].id();
    if(abs(myid) < 6 || abs(myid) == 21) continue;
    last = ip;
  }
  if(Verbosity)
    cout << "last before powheg emission = " << last << " , id = "
         << e[last].id() << " emission size = " << e.size() - 1 - last << endl;

  // Some events may not have radiation from POWHEG
  switch (e.size() - 1 - last) {
  case 0:
    // No radiation - veto scale is given by the factorisation scale
    pTpowheg = -1.;
    pTveto   = infoPtr->QFac();
    noRad    = true;

    // If this is the first no radiation event, then print scale
    if (firstNoRad) {
      if(Verbosity)
        cout << "Info: no POWHEG radiation, Q = " << pTveto
             << " GeV" << endl;
      firstNoRad = false;
    }
    break;

  case 1:
    // Radiation is parton last+1 - first check that it is as expected
    if (e[last+1].id() != 21 && e[last+1].idAbs() > 5) {
      cout << endl << "Emergency dump of the intermediate event: " << endl;
      e.list();
      fatalEmissionVeto(string("Error: jet is not quark/gluon"));
    }
    // Veto scale is given by jet pT
    pTpowheg = e[last+1].pT();
    pTveto = e[last+1].pT();
    noRad  = false;
    break;
  }

  if(Verbosity) cout << "veto pT = " << pTveto << " QFac = " << infoPtr->QFac() << endl;

  // Initialise other variables
  pTshower = -1.;

  // Do not veto the event
  return false;
}

// For subsequent ISR/FSR emissions, find the pT of the shower
// emission and veto as necessary
bool EmissionVetoHook::doVetoISREmission(int, const Pythia8::Event &e, int iSys) {
  // Must be radiation from the hard system
  if (iSys != 0) return false;

  if(last < 0) fatalEmissionVeto(string("Variable last is not filled"));

  // ISR - next shower emission is given status 43
  int i;
  for (i = e.size() - 1; i > last; i--)
    if (e[i].isFinal() && e[i].status() == 43) break;
  if (i == last) {
    cout << endl << "Emergency dump of the intermediate event: " << endl;
    e.list();
    fatalEmissionVeto(string("Error: couldn't find ISR emission"));
  }
     
  // Veto if above the POWHEG scale
  if (e[i].pT() > pTveto) {
    nISRveto++;
    if(Verbosity) cout << "ISR vetoed" << endl;
    return true; 
  }
  // Store the first shower emission pT
  if (pTshower < 0.) pTshower = e[i].pT();
   
  return false;
}

bool EmissionVetoHook::doVetoFSREmission(int, const Pythia8::Event &e, int iSys, bool) {
  // Must be radiation from the hard system
  if (iSys != 0) return false;

  // FSR - shower emission will have status 51 and not be t/tbar
  int i;
  for (i = e.size() - 1; i > last; i--)
    if (e[i].isFinal() && e[i].status() == 51 &&
        e[i].idAbs() != 6) break;
  if (i == last) {
    cout << endl << "Emergency dump of the intermediate event: " << endl;
    e.list();
    fatalEmissionVeto(string("Error: couldn't find FSR emission"));
  }

  // Veto if above the POWHEG scale
  if (e[i].pT() > pTveto) {
    nFSRveto++;  
    if(Verbosity) cout << "FSR vetoed" << endl;
    return true;
  }
  // Store the first shower emission pT   
  if (pTshower < 0.) pTshower = e[i].pT();

  return false;
}
