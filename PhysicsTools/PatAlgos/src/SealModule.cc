
#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/PatAlgos/interface/PATElectronProducer.h"
#include "PhysicsTools/PatAlgos/interface/PATMuonProducer.h"
#include "PhysicsTools/PatAlgos/interface/PATTauProducer.h"
#include "PhysicsTools/PatAlgos/interface/PATJetProducer.h"
#include "PhysicsTools/PatAlgos/interface/PATMETProducer.h"

DEFINE_FWK_MODULE(pat::PATElectronProducer);
DEFINE_FWK_MODULE(pat::PATMuonProducer);
DEFINE_FWK_MODULE(pat::PATTauProducer);
DEFINE_FWK_MODULE(pat::PATJetProducer);
DEFINE_FWK_MODULE(pat::PATMETProducer);


#include "PhysicsTools/PatAlgos/interface/PATObjectSelector.h"

DEFINE_FWK_MODULE(pat::PATElectronSelector);
DEFINE_FWK_MODULE(pat::PATMuonSelector);
DEFINE_FWK_MODULE(pat::PATTauSelector);
DEFINE_FWK_MODULE(pat::PATJetSelector);
DEFINE_FWK_MODULE(pat::PATMETSelector);
DEFINE_FWK_MODULE(pat::PATParticleSelector);


#include "PhysicsTools/PatAlgos/interface/PATLeptonCountFilter.h"

DEFINE_FWK_MODULE(pat::PATLeptonCountFilter);

#include "PhysicsTools/PatAlgos/interface/PATObjectFilter.h"

DEFINE_FWK_MODULE(pat::PATElectronMinFilter);
DEFINE_FWK_MODULE(pat::PATMuonMinFilter);
DEFINE_FWK_MODULE(pat::PATTauMinFilter);
DEFINE_FWK_MODULE(pat::PATJetMinFilter);
DEFINE_FWK_MODULE(pat::PATMETMinFilter);
DEFINE_FWK_MODULE(pat::PATParticleMinFilter);

DEFINE_FWK_MODULE(pat::PATElectronMaxFilter);
DEFINE_FWK_MODULE(pat::PATMuonMaxFilter);
DEFINE_FWK_MODULE(pat::PATTauMaxFilter);
DEFINE_FWK_MODULE(pat::PATJetMaxFilter);
DEFINE_FWK_MODULE(pat::PATMETMaxFilter);
DEFINE_FWK_MODULE(pat::PATParticleMaxFilter);


#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"

#include "PhysicsTools/PatAlgos/interface/ObjectEnergyScale.h"

namespace pat {
  typedef ObjectEnergyScale<pat::Electron> ElectronEnergyScale;
  typedef ObjectEnergyScale<pat::Muon>     MuonEnergyScale;
  typedef ObjectEnergyScale<pat::Tau>      TauEnergyScale;
  typedef ObjectEnergyScale<pat::Jet>      JetEnergyScale;
  typedef ObjectEnergyScale<pat::MET>      METEnergyScale;
}

DEFINE_FWK_MODULE(pat::ElectronEnergyScale);
DEFINE_FWK_MODULE(pat::MuonEnergyScale);
DEFINE_FWK_MODULE(pat::TauEnergyScale);
DEFINE_FWK_MODULE(pat::JetEnergyScale);
DEFINE_FWK_MODULE(pat::METEnergyScale);

#include "PhysicsTools/PatAlgos/interface/ObjectSpatialResolution.h"

namespace pat {
  typedef ObjectSpatialResolution<pat::Electron> ElectronSpatialResolution;
  typedef ObjectSpatialResolution<pat::Muon>     MuonSpatialResolution;
  typedef ObjectSpatialResolution<pat::Tau>      TauSpatialResolution;
  typedef ObjectSpatialResolution<pat::Jet>      JetSpatialResolution;
  typedef ObjectSpatialResolution<pat::MET>      METSpatialResolution;
}

DEFINE_FWK_MODULE(pat::ElectronSpatialResolution);
DEFINE_FWK_MODULE(pat::MuonSpatialResolution);
DEFINE_FWK_MODULE(pat::TauSpatialResolution);
DEFINE_FWK_MODULE(pat::JetSpatialResolution);
DEFINE_FWK_MODULE(pat::METSpatialResolution);


// PLEASE DONT MOVE THE CLEANERS ABOVE THE OBJECT SELECTORS
// There is a problem with the definition of helper namespace
// I'll check it in more detail later
//             Giovanni
#include "PhysicsTools/PatAlgos/interface/PATElectronCleaner.h"
#include "PhysicsTools/PatAlgos/interface/PATMuonCleaner.h"
#include "PhysicsTools/PatAlgos/interface/PATTauCleaner.h"
#include "PhysicsTools/PatAlgos/interface/PATTauCleaner.icc"
DEFINE_FWK_MODULE(pat::PATElectronCleaner);
DEFINE_FWK_MODULE(pat::PATMuonCleaner);
DEFINE_FWK_MODULE(pat::PATPFTauCleaner);
DEFINE_FWK_MODULE(pat::PATPFTau2BaseCleaner);
DEFINE_FWK_MODULE(pat::PATCaloTauCleaner);
DEFINE_FWK_MODULE(pat::PATCaloTau2BaseCleaner);


