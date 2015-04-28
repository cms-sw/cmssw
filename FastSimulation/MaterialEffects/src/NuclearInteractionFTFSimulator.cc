// Framework Headers
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
#include "G4CascadeInterface.hh"

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
#include "G4IonTable.hh"
#include "G4ProcessManager.hh"
#include "G4PhysicsLogVector.hh"
#include "G4SystemOfUnits.hh"

const G4ParticleDefinition* NuclearInteractionFTFSimulator::theG4Hadron[] = {0};
int NuclearInteractionFTFSimulator::theId[] = {0};

const double fact = 1.0/CLHEP::GeV;

// interaction length corrections per particle and energy 
const double corrfactors[numHadrons][npoints] = {
  {1.0872, 1.1026, 1.111, 1.111, 1.0105, 0.97622, 0.9511, 0.9526, 0.97591, 0.99277, 1.0099, 1.015, 1.0217, 1.0305, 1.0391, 1.0438, 1.0397, 1.0328, 1.0232, 1.0123, 1.0},
  {1.0416, 1.1044, 1.1467, 1.1273, 1.026, 0.99085, 0.96572, 0.96724, 0.99091, 1.008, 1.0247, 1.0306, 1.0378, 1.0427, 1.0448, 1.0438, 1.0397, 1.0328, 1.0232, 1.0123, 1.0},
  {0.5308, 0.53589, 0.67059, 0.80253, 0.82341, 0.79083, 0.85967, 0.90248, 0.93792, 0.9673, 1.0034, 1.022, 1.0418, 1.0596, 1.0749, 1.079, 1.0704, 1.0576, 1.0408, 1.0214, 1.0},
  {0.49107, 0.50571, 0.64149, 0.77209, 0.80472, 0.78166, 0.83509, 0.8971, 0.93234, 0.96154, 0.99744, 1.0159, 1.0355, 1.0533, 1.0685, 1.0732, 1.0675, 1.0485, 1.0355, 1.0191, 1.0},
  {1.9746, 1.7887, 1.5645, 1.2817, 1.0187, 0.95216, 0.9998, 1.035, 1.0498, 1.0535, 1.0524, 1.0495, 1.0461, 1.0424, 1.0383, 1.0338, 1.0287, 1.0228, 1.0161, 1.0085, 1.0},
  {0.46028, 0.59514, 0.70355, 0.70698, 0.62461, 0.65103, 0.71945, 0.77753, 0.83582, 0.88422, 0.92117, 0.94889, 0.96963, 0.98497, 0.99596, 1.0033, 1.0075, 1.0091, 1.0081, 1.005, 1.0},
  {0.75016, 0.89607, 0.97185, 0.91083, 0.77425, 0.77412, 0.8374, 0.88848, 0.93104, 0.96174, 0.98262, 0.99684, 1.0065, 1.0129, 1.0168, 1.0184, 1.018, 1.0159, 1.0121, 1.0068, 1.0},
  {0.75016, 0.89607, 0.97185, 0.91083, 0.77425, 0.77412, 0.8374, 0.88848, 0.93104, 0.96174, 0.98262, 0.99684, 1.0065, 1.0129, 1.0168, 1.0184, 1.018, 1.0159, 1.0121, 1.0068, 1.0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0},
  {1.1006, 1.1332, 1.121, 1.1008, 1.086, 1.077, 1.0717, 1.0679, 1.0643, 1.0608, 1.057, 1.053, 1.0487, 1.0441, 1.0392, 1.0338, 1.028, 1.0218, 1.015, 1.0078, 1.0},
  {1.1318, 1.1255, 1.1062, 1.0904, 1.0802, 1.0742, 1.0701, 1.0668, 1.0636, 1.0602, 1.0566, 1.0527, 1.0485, 1.044, 1.0391, 1.0337, 1.028, 1.0217, 1.015, 1.0078, 1.0},
  {1.1094, 1.1332, 1.1184, 1.0988, 1.0848, 1.0765, 1.0714, 1.0677, 1.0642, 1.0607, 1.0569, 1.053, 1.0487, 1.0441, 1.0391, 1.0338, 1.028, 1.0218, 1.015, 1.0078, 1.0},
  {1.1087, 1.1332, 1.1187, 1.099, 1.0849, 1.0765, 1.0715, 1.0677, 1.0642, 1.0607, 1.057, 1.053, 1.0487, 1.0441, 1.0391, 1.0338, 1.028, 1.0218, 1.015, 1.0078, 1.0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0},
  {1.1192, 1.132, 1.1147, 1.0961, 1.0834, 1.0758, 1.0711, 1.0674, 1.064, 1.0606, 1.0569, 1.0529, 1.0486, 1.0441, 1.0391, 1.0338, 1.028, 1.0218, 1.015, 1.0078, 1.0},
  {1.1188, 1.1321, 1.1149, 1.0963, 1.0834, 1.0758, 1.0711, 1.0675, 1.0641, 1.0606, 1.0569, 1.0529, 1.0486, 1.0441, 1.0391, 1.0338, 1.028, 1.0218, 1.015, 1.0078, 1.0},
  {0.50776, 0.5463, 0.5833, 0.61873, 0.65355, 0.68954, 0.72837, 0.7701, 0.81267, 0.85332, 0.89037, 0.92329, 0.95177, 0.97539, 0.99373, 1.0066, 1.014, 1.0164, 1.0144, 1.0087, 1.0},
  {0.50787, 0.5464, 0.58338, 0.6188, 0.65361, 0.6896, 0.72841, 0.77013, 0.8127, 0.85333, 0.89038, 0.92329, 0.95178, 0.9754, 0.99373, 1.0066, 1.014, 1.0164, 1.0144, 1.0087, 1.0},
  {1.1006, 1.1332, 1.121, 1.1008, 1.086, 1.077, 1.0717, 1.0679, 1.0643, 1.0608, 1.057, 1.053, 1.0487, 1.0441, 1.0392, 1.0338, 1.028, 1.0218, 1.015, 1.0078, 1.0},
  {1.1318, 1.1255, 1.1062, 1.0904, 1.0802, 1.0742, 1.0701, 1.0668, 1.0636, 1.0602, 1.0566, 1.0527, 1.0485, 1.044, 1.0391, 1.0337, 1.028, 1.0217, 1.015, 1.0078, 1.0},
  {1.1094, 1.1332, 1.1184, 1.0988, 1.0848, 1.0765, 1.0714, 1.0677, 1.0642, 1.0607, 1.0569, 1.053, 1.0487, 1.0441, 1.0391, 1.0338, 1.028, 1.0218, 1.015, 1.0078, 1.0},
  {1.1087, 1.1332, 1.1187, 1.099, 1.0849, 1.0765, 1.0715, 1.0677, 1.0642, 1.0607, 1.057, 1.053, 1.0487, 1.0441, 1.0391, 1.0338, 1.028, 1.0218, 1.015, 1.0078, 1.0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0},
  {1.1192, 1.132, 1.1147, 1.0961, 1.0834, 1.0758, 1.0711, 1.0674, 1.064, 1.0606, 1.0569, 1.0529, 1.0486, 1.0441, 1.0391, 1.0338, 1.028, 1.0218, 1.015, 1.0078, 1.0},
  {1.1188, 1.1321, 1.1149, 1.0963, 1.0834, 1.0758, 1.0711, 1.0675, 1.0641, 1.0606, 1.0569, 1.0529, 1.0486, 1.0441, 1.0391, 1.0338, 1.028, 1.0218, 1.015, 1.0078, 1.0},
  {0.47677, 0.51941, 0.56129, 0.60176, 0.64014, 0.67589, 0.70891, 0.73991, 0.77025, 0.80104, 0.83222, 0.86236, 0.8901, 0.91518, 0.9377, 0.95733, 0.97351, 0.98584, 0.9942, 0.99879, 1.0},
  {0.49361, 0.53221, 0.56976, 0.60563, 0.63954, 0.67193, 0.70411, 0.73777, 0.77378, 0.81114, 0.84754, 0.88109, 0.91113, 0.93745, 0.95974, 0.97762, 0.99081, 0.99929, 1.0033, 1.0034, 1.0},
  {0.4873, 0.52744, 0.56669, 0.60443, 0.64007, 0.67337, 0.70482, 0.73572, 0.76755, 0.80086, 0.83456, 0.86665, 0.8959, 0.92208, 0.94503, 0.96437, 0.97967, 0.99072, 0.99756, 1.0005, 1.0},
  {0.48729, 0.52742, 0.56668, 0.60442, 0.64006, 0.67336, 0.70482, 0.73571, 0.76754, 0.80086, 0.83455, 0.86665, 0.8959, 0.92208, 0.94503, 0.96437, 0.97967, 0.99072, 0.99756, 1.0005, 1.0},
};

// interaction length in Silicon at 1 TeV per particle
const double nuclIntLength[numHadrons] = {
4.5606, 4.4916, 5.7511, 5.7856, 6.797, 6.8373, 6.8171, 6.8171, 0, 0, 4.6926, 4.6926, 4.6926, 4.6926, 0, 4.6926, 4.6926, 4.3171, 4.3171, 4.6926, 4.6926, 4.6926, 4.6926, 0, 4.6926, 4.6926, 2.509, 2.9048, 2.5479, 2.5479
};

NuclearInteractionFTFSimulator::NuclearInteractionFTFSimulator(  
  unsigned int distAlgo, double distCut, double elimit, double eth) :
  curr4Mom(0.,0.,0.,0.),
  vectProj(0.,0.,1.),
  theBoost(0.,0.,0.),
  theBertiniLimit(elimit),
  theEnergyLimit(eth),
  theDistCut(distCut),
  distMin(1E99),
  theDistAlgo(distAlgo)
{
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

  // Bertini Cascade 
  theBertiniCascade = new G4CascadeInterface();

  // Geant4 particles and cross sections
  if(!theG4Hadron[0]) {
    theG4Hadron[0] = G4Proton::Proton();
    theG4Hadron[1] = G4Neutron::Neutron();
    theG4Hadron[2] = G4PionPlus::PionPlus();
    theG4Hadron[3] = G4PionMinus::PionMinus();
    theG4Hadron[4] = G4KaonPlus::KaonPlus();
    theG4Hadron[5] = G4KaonMinus::KaonMinus();
    theG4Hadron[6] = G4KaonZeroLong::KaonZeroLong();
    theG4Hadron[7] = G4KaonZeroShort::KaonZeroShort();
    theG4Hadron[8] = G4KaonZero::KaonZero();
    theG4Hadron[9] = G4AntiKaonZero::AntiKaonZero();
    theG4Hadron[10]= G4Lambda::Lambda();
    theG4Hadron[11]= G4OmegaMinus::OmegaMinus();
    theG4Hadron[12]= G4SigmaMinus::SigmaMinus();
    theG4Hadron[13]= G4SigmaPlus::SigmaPlus();
    theG4Hadron[14]= G4SigmaZero::SigmaZero();
    theG4Hadron[15]= G4XiMinus::XiMinus();
    theG4Hadron[16]= G4XiZero::XiZero();
    theG4Hadron[17]= G4AntiProton::AntiProton();
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

    // other Geant4 particles
    G4ParticleDefinition* ion = G4GenericIon::GenericIon();
    ion->SetProcessManager(new G4ProcessManager(ion));
    G4DecayPhysics decays;
    decays.ConstructParticle();  
    G4ParticleTable* partTable = G4ParticleTable::GetParticleTable();
    partTable->SetVerboseLevel(0);
    partTable->SetReadiness();

    for(int i=0; i<numHadrons; ++i) {
      theId[i] = theG4Hadron[i]->GetPDGEncoding();
    }
  }

  // local objects
  vect = new G4PhysicsLogVector(npoints-1,100*MeV,TeV);
  currIdx = 0;
  index = 0;
  currTrack = 0;
  currParticle = theG4Hadron[0];

  // fill projectile particle definitions
  dummyStep = new G4Step();
  dummyStep->SetPreStepPoint(new G4StepPoint());

  // target is always Silicon
  targetNucleus.SetParameters(28, 14);
}

NuclearInteractionFTFSimulator::~NuclearInteractionFTFSimulator() {

  delete theStringDecay;
  delete theStringModel;
  delete theLund;
  delete vect;
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
  double mass = currParticle->GetPDGMass();
  double ekin = CLHEP::GeV*Particle.momentum().e() - mass;
  double ff;
  if(ekin <= vect->Energy(0)) {
    ff = corrfactors[currIdx][0];
  } else if(ekin >= vect->Energy(npoints-1)) {
    ff = 1.0;
  } else {
    index = vect->FindBin(ekin, index);
    double e1 = vect->Energy(index);
    double e2 = vect->Energy(index+1);
    ff = (corrfactors[currIdx][index]*(e2 - ekin) + 
	  corrfactors[currIdx][index+1]*(ekin - e1))/(e2 - e1);
  }
  /*
  std::cout << " Primary " <<  currParticle->GetParticleName() 
  	    << "  E(GeV)= " << e*fact << std::endl;
  */

  double currInteractionLength = -G4Log(random->flatShoot())*nuclIntLength[currIdx]*ff; 
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

  G4DynamicParticle* dynParticle = new G4DynamicParticle(theG4Hadron[currIdx],dir,ekin);
  currTrack = new G4Track(dynParticle, 0.0, vectProj);
  currTrack->SetStep(dummyStep);

  theProjectile.Initialise(*currTrack); 
  delete currTrack;

  G4HadFinalState* result;
  // Bertini cascade for low-energy hadrons (except light anti-nuclei)
  // FTFP is applied above energy limit and for all anti-hyperons and anti-ions 
  if(ekin <= theBertiniLimit && currIdx < 17) { 
    result = theBertiniCascade->ApplyYourself(theProjectile, targetNucleus);
  } else {
    result = theHadronicModel->ApplyYourself(theProjectile, targetNucleus);
  }
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
          if(curr4Mom.e() > theEnergyLimit) { 
	    saveDaughter(Particle, curr4Mom, 22); 
	  } 
	} else {
          if(curr4Mom.e() > theEnergyLimit + dp->GetParticleDefinition()->GetPDGMass()) { 
	    saveDaughter(Particle, curr4Mom, thePid); 
	  }
	}
      }
      result->Clear();
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
