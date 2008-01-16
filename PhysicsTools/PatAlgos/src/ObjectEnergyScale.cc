
#include "FWCore/Framework/interface/MakerMacros.h"

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

