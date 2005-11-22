#include "FWCore/EDProduct/interface/Wrapper.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include <vector>

namespace {
  namespace {
    HepGeom::Vector3D<float> v3d;
    HepGeom::Point3D<float> p3d;
    std::vector<reco::Muon> v1;
    edm::Wrapper<std::vector<reco::Muon> > c1;
  }
}
