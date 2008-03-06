#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "PhysicsTools/PatAlgos/plugins/ObjectEnergyScale.h"

namespace pat {
  typedef ObjectEnergyScale<pat::Electron> ElectronEnergyScale;
  typedef ObjectEnergyScale<pat::Muon>     MuonEnergyScale;
  typedef ObjectEnergyScale<pat::Tau>      TauEnergyScale;
  typedef ObjectEnergyScale<pat::Jet>      JetEnergyScale;
  typedef ObjectEnergyScale<pat::MET>      METEnergyScale;
}

using namespace pat;
DEFINE_FWK_MODULE(ElectronEnergyScale);
DEFINE_FWK_MODULE(MuonEnergyScale);
DEFINE_FWK_MODULE(TauEnergyScale);
DEFINE_FWK_MODULE(JetEnergyScale);
DEFINE_FWK_MODULE(METEnergyScale);

