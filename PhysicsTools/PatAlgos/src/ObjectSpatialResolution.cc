
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"

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


// the cleaners get their modules made inside their own cc files
