#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "PhysicsTools/PatAlgos/plugins/ObjectSpatialResolution.h"

namespace pat {
  typedef ObjectSpatialResolution<pat::Electron> ElectronSpatialResolution;
  typedef ObjectSpatialResolution<pat::Muon>     MuonSpatialResolution;
  typedef ObjectSpatialResolution<pat::Tau>      TauSpatialResolution;
  typedef ObjectSpatialResolution<pat::Jet>      JetSpatialResolution;
  typedef ObjectSpatialResolution<pat::MET>      METSpatialResolution;
}

using namespace pat;
DEFINE_FWK_MODULE(ElectronSpatialResolution);
DEFINE_FWK_MODULE(MuonSpatialResolution);
DEFINE_FWK_MODULE(TauSpatialResolution);
DEFINE_FWK_MODULE(JetSpatialResolution);
DEFINE_FWK_MODULE(METSpatialResolution);


