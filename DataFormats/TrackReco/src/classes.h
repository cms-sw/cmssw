#include "FWCore/EDProduct/interface/Wrapper.h"
#include "DataFormats/TrackReco/interface/HelixParameters.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/RecHit.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include <vector>

namespace {
  namespace {
    HepGeom::Vector3D<double> v0;
    HepGeom::Point3D<double> p0;

    reco::Vector<5> v5_1;
    reco::Error<5> e5_1;

    std::vector<reco::Track> v1;
    edm::Wrapper<std::vector<reco::Track> > c1;

    std::vector<reco::RecHit> v2;
    edm::Wrapper<std::vector<reco::RecHit> > c2;

    reco::Vector<5> v3_3;
    edm::Ref<std::vector<reco::RecHit> > r3;
    edm::RefVector<std::vector<reco::RecHit> > rv3;
    std::vector<reco::TrackExtra> v3;
    edm::Wrapper<std::vector<reco::TrackExtra> > c3;
  }
}
