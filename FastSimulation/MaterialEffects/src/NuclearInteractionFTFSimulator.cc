// Framework Headers
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Fast Sim headers
#include "FastSimulation/MaterialEffects/interface/NuclearInteractionFTFSimulator.h"
#include "FastSimulation/MaterialEffects/interface/CMSDummyDeexcitation.h"
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"

// Geant4 headers
#include "G4ParticleDefinition.hh"
#include "G4DynamicParticle.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4TheoFSGenerator.hh"
#include "G4FTFModel.hh"
#include "G4ExcitedStringDecay.hh"
#include "G4LundStringFragmentation.hh"
#include "G4GeneratorPrecompoundInterface.hh"

#include "G4Proton.hh"
#include "G4Neutron.hh"
#include "G4PionPlus.hh"
#include "G4PionMinus.hh"
#include "G4AntiProton.hh"
#include "G4AntiNeutron.hh"
#include "G4KaonPlus.hh"
#include "G4KaonMinus.hh"
#include "G4KaonZeroLong.hh"
#include "G4KaonZeroShort.hh"
#include "G4KaonZero.hh"
#include "G4AntiKaonZero.hh"
#include "G4GenericIon.hh"

#include "G4Lambda.hh"
#include "G4OmegaMinus.hh"
#include "G4SigmaMinus.hh"
#include "G4SigmaPlus.hh"
#include "G4SigmaZero.hh"
#include "G4XiMinus.hh"
#include "G4XiZero.hh"
#include "G4AntiLambda.hh"
#include "G4AntiOmegaMinus.hh"
#include "G4AntiSigmaMinus.hh"
#include "G4AntiSigmaPlus.hh"
#include "G4AntiSigmaZero.hh"
#include "G4AntiXiMinus.hh"
#include "G4AntiXiZero.hh"
#include "G4AntiAlpha.hh"
#include "G4AntiDeuteron.hh"
#include "G4AntiTriton.hh"
#include "G4AntiHe3.hh"

#include "G4Material.hh"
#include "G4DecayPhysics.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"

const double fact = 1.0/CLHEP::GeV;

NuclearInteractionFTFSimulator::NuclearInteractionFTFSimulator(  
  unsigned int distAlgo, 
  double distCut) :
  theDistCut(distCut),
  distMin(1E99),
  theDistAlgo(distAlgo)
{
  theEnergyLimit = 1*CLHEP::GeV;
  currIdx = 0;

  // FTF model
  theHadronicModel = new G4TheoFSGenerator("FTF");
  theStringModel = new G4FTFModel();
  G4GeneratorPrecompoundInterface* cascade 
    = new G4GeneratorPrecompoundInterface(new CMSDummyDeexcitation());
  theLund = new G4LundStringFragmentation();
  theStringDecay = new G4ExcitedStringDecay(theLund);
  theStringModel->SetFragmentationModel(theStringDecay);

  theHadronicModel->SetTransport(cascade);
  theHadronicModel->SetHighEnergyGenerator(theStringModel);
  theHadronicModel->SetMinEnergy(theEnergyLimit);

  // Geant4 particles
  numHadrons = 30;
  theG4Hadron.resize(numHadrons,0);
  theG4Hadron[0] = G4Proton::Proton();
  theG4Hadron[1] = G4Neutron::Neutron();
  theG4Hadron[2] = G4PionPlus::PionPlus();
  theG4Hadron[3] = G4PionMinus::PionMinus();
  theG4Hadron[4] = G4AntiProton::AntiProton();
  theG4Hadron[5] = G4KaonPlus::KaonPlus();
  theG4Hadron[6] = G4KaonMinus::KaonMinus();
  theG4Hadron[7] = G4KaonZeroLong::KaonZeroLong();
  theG4Hadron[8] = G4KaonZeroShort::KaonZeroShort();
  theG4Hadron[9] = G4KaonZero::KaonZero();
  theG4Hadron[10]= G4AntiKaonZero::AntiKaonZero();
  theG4Hadron[11]= G4Lambda::Lambda();
  theG4Hadron[12]= G4OmegaMinus::OmegaMinus();
  theG4Hadron[13]= G4SigmaMinus::SigmaMinus();
  theG4Hadron[14]= G4SigmaPlus::SigmaPlus();
  theG4Hadron[15]= G4SigmaZero::SigmaZero();
  theG4Hadron[16]= G4XiMinus::XiMinus();
  theG4Hadron[17]= G4XiZero::XiZero();
  theG4Hadron[18]= G4AntiNeutron::AntiNeutron();
  theG4Hadron[19]= G4AntiLambda::AntiLambda();
  theG4Hadron[20]= G4AntiOmegaMinus::AntiOmegaMinus();
  theG4Hadron[21]= G4AntiSigmaMinus::AntiSigmaMinus();
  theG4Hadron[22]= G4AntiSigmaPlus::AntiSigmaPlus();
  theG4Hadron[23]= G4AntiSigmaZero::AntiSigmaZero();
  theG4Hadron[24]= G4AntiXiMinus::AntiXiMinus();
  theG4Hadron[25]= G4AntiXiZero::AntiXiZero();
  theG4Hadron[26]= G4AntiAlpha::AntiAlpha();
  theG4Hadron[27]= G4AntiDeuteron::AntiDeuteron();
  theG4Hadron[28]= G4AntiTriton::AntiTriton();
  theG4Hadron[29]= G4AntiHe3::AntiHe3();

  G4GenericIon::GenericIon();
  G4DecayPhysics decays;
  decays.ConstructParticle();  
  G4ParticleTable* partTable = G4ParticleTable::GetParticleTable();
  partTable->SetReadiness();

  // interaction length in units of radiation length 
  // computed for 5 GeV projectile energy, default value for K0_L
  theNuclIntLength.resize(numHadrons,6.46);
  theNuclIntLength[0] = 4.528;
  theNuclIntLength[1] = 4.524;
  theNuclIntLength[2] = 4.493;
  theNuclIntLength[3] = 4.493;
  theNuclIntLength[4] = 3.593;
  theNuclIntLength[5] = 7.154;
  theNuclIntLength[6] = 5.889;
  theNuclIntLength[11]= 4.986;
  theNuclIntLength[12]= 4.983;
  theNuclIntLength[13]= 4.986;
  theNuclIntLength[14]= 4.986;
  theNuclIntLength[15]= 4.986;
  theNuclIntLength[16]= 4.986;
  theNuclIntLength[17]= 4.986;
  theNuclIntLength[18]= 3.597;
  theNuclIntLength[19]= 3.608;
  theNuclIntLength[20]= 3.639;
  theNuclIntLength[21]= 3.613;
  theNuclIntLength[22]= 3.613;
  theNuclIntLength[23]= 3.613;
  theNuclIntLength[24]= 3.62;
  theNuclIntLength[25]= 3.62;
  theNuclIntLength[26]= 1.971;
  theNuclIntLength[27]= 2.301;
  theNuclIntLength[28]= 1.997;
  theNuclIntLength[29]= 1.997;

  // list of PDG codes
  theId.resize(numHadrons,0);

  // local objects
  currIdx = 0;
  currTrack = 0;
  currParticle = theG4Hadron[0];
  vectProj.set(0.0,0.0,1.0);  
  theBoost.set(0.0,0.0,1.0);  

  // fill projectile particle definitions
  dummyStep = new G4Step();
  dummyStep->SetPreStepPoint(new G4StepPoint());
  for(int i=0; i<numHadrons; ++i) {
    theId[i] = theG4Hadron[i]->GetPDGEncoding();
  }

  // target is always Silicon
  targetNucleus.SetParameters(28, 14);
}

NuclearInteractionFTFSimulator::~NuclearInteractionFTFSimulator() {

  delete theStringDecay;
  delete theStringModel;
  delete theLund;
}

void NuclearInteractionFTFSimulator::compute(ParticlePropagator& Particle, 
					     RandomEngineAndDistribution const* random)
{
  //std::cout << "#### Primary " << Particle.pid() << " E(GeV)= " 
  //	    << Particle.momentum().e() << std::endl;

  int thePid = Particle.pid(); 
  if(thePid != theId[currIdx]) {
    currParticle = 0;
    currIdx = 0;
    for(; currIdx<numHadrons; ++currIdx) {
      if(theId[currIdx] == thePid) {
	currParticle = theG4Hadron[currIdx];
	// neutral kaons
	if(7 == currIdx || 8 == currIdx) {
	  currParticle = theG4Hadron[9];
	  if(random->flatShoot() > 0.5) { currParticle = theG4Hadron[10]; }
	}
	break;
      }
    }
  }
  if(!currParticle) { return; }

  // fill projectile for Geant4
  double e = CLHEP::GeV*Particle.momentum().e();
  double mass = currParticle->GetPDGMass();
  /*
  std::cout << " Primary " <<  currParticle->GetParticleName() 
  	    << "  E(GeV)= " << e*fact << std::endl;
  */
  if(e <= theEnergyLimit + mass) { return; }

  double  currInteractionLength = -G4Log(random->flatShoot())*theNuclIntLength[currIdx]; 
  /*
  std::cout << "*NuclearInteractionFTFSimulator::compute: R(X0)= " << radLengths
	    << " Rnuc(X0)= " << theNuclIntLength[currIdx] << "  IntLength(X0)= " 
            << currInteractionLength << std::endl;
  */
  // Check position of nuclear interaction
  if (currInteractionLength > radLengths) { return; }

  // fill projectile for Geant4
  double px = Particle.momentum().px();
  double py = Particle.momentum().py();
  double pz = Particle.momentum().pz();
  double norm = 1.0/sqrt(px*px + py*py + pz*pz);
  G4ThreeVector dir(px*norm, py*norm, pz*norm);
  /*
  std::cout << " Primary " <<  currParticle->GetParticleName() 
	    << "  E(GeV)= " << e*fact << "  P(GeV/c)= (" 
	    << px << " " << py << " " << pz << ")" << std::endl;
  */

  G4DynamicParticle* dynParticle = new G4DynamicParticle(theG4Hadron[currIdx],dir,e-mass);
  currTrack = new G4Track(dynParticle, 0.0, vectProj);
  currTrack->SetStep(dummyStep);

  theProjectile.Initialise(*currTrack); 
  delete currTrack;

  G4HadFinalState* result = theHadronicModel->ApplyYourself(theProjectile, targetNucleus);

  if(result) {

    int nsec = result->GetNumberOfSecondaries();
    if(0 < nsec) {

      result->SetTrafoToLab(theProjectile.GetTrafoToLab());
      _theUpdatedState.clear();

      //std::cout << "   " << nsec << " secondaries" << std::endl;
      // Generate angle
      double phi = random->flatShoot()*CLHEP::twopi;
      theClosestChargedDaughterId = -1;
      distMin = 1e99;

      // rotate and store secondaries
      for (int j=0; j<nsec; ++j) {

        const G4DynamicParticle* dp = result->GetSecondary(j)->GetParticle();
        int thePid = dp->GetParticleDefinition()->GetPDGEncoding();

	// rotate around primary direction
	curr4Mom = dp->Get4Momentum();
	curr4Mom.rotate(phi, vectProj);
	curr4Mom *= result->GetTrafoToLab();
	/*
	std::cout << j << ". " << dp->GetParticleDefinition()->GetParticleName() 
		  << "  " << thePid
		  << "  " << curr4Mom*fact << std::endl;
	*/
	// prompt 2-gamma decay for pi0, eta, eta'p
        if(111 == thePid || 221 == thePid || 331 == thePid) {
          theBoost = curr4Mom.boostVector();
          double e = 0.5*dp->GetParticleDefinition()->GetPDGMass();
          double fi  = random->flatShoot()*CLHEP::twopi; 
          double cth = 2*random->flatShoot() - 1.0;
          double sth = sqrt((1.0 - cth)*(1.0 + cth)); 
          G4LorentzVector lv(e*sth*cos(fi),e*sth*sin(fi),e*cth,e);
          lv.boost(theBoost);
	  saveDaughter(Particle, lv, 22); 
          curr4Mom -= lv;
	  saveDaughter(Particle, curr4Mom, 22); 
	} else {
	  saveDaughter(Particle, curr4Mom, thePid); 
	}
      }
    }
  }
}

void NuclearInteractionFTFSimulator::saveDaughter(ParticlePropagator& Particle, 
						  const G4LorentzVector& lv, int pdgid)
{
  unsigned int idx = _theUpdatedState.size();   
  _theUpdatedState.push_back(Particle);
  _theUpdatedState[idx].SetXYZT(lv.px()*fact,lv.py()*fact,lv.pz()*fact,lv.e()*fact);
  _theUpdatedState[idx].setID(pdgid);

  // Store the closest daughter index (for later tracking purposes, so charged particles only) 
  double distance = distanceToPrimary(Particle,_theUpdatedState[idx]);
  // Find the closest daughter, if closer than a given upper limit.
  if ( distance < distMin && distance < theDistCut ) {
    distMin = distance;
    theClosestChargedDaughterId = idx;
  }
  // std::cout << _theUpdatedState[idx] << std::endl;
}

double 
NuclearInteractionFTFSimulator::distanceToPrimary(const RawParticle& Particle,
						  const RawParticle& aDaughter) const 
{
  double distance = 2E99;
  // Compute the distance only for charged primaries
  if ( fabs(Particle.charge()) > 1E-6 ) { 

    // The secondary must have the same charge
    double chargeDiff = fabs(aDaughter.charge()-Particle.charge());
    if ( fabs(chargeDiff) < 1E-6 ) {

      // Here are two distance definitions * to be tuned *
      switch ( theDistAlgo ) { 
	
      case 1:
	// sin(theta12)
	distance = (aDaughter.Vect().Unit().Cross(Particle.Vect().Unit())).R();
	break;
	
      case 2: 
	// sin(theta12) * p1/p2
	distance = (aDaughter.Vect().Cross(Particle.Vect())).R()
	  /aDaughter.Vect().Mag2();
	break;
	
      default:
	// Should not happen
	break;	
      }
    }
  } 
  return distance;
}
