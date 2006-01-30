#include "FWCore/EDProduct/interface/Wrapper.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <vector>

namespace {
  namespace {
    reco::Vertex::TrackRefs rvs1;
    std::vector<reco::Vertex> v1;
    reco::VertexCollection vv1;
    edm::Wrapper<reco::VertexCollection> wc1;
    edm::RefProd<reco::VertexCollection> rp1;
    edm::Ref<reco::VertexCollection> r1;
    edm::RefVector<reco::VertexCollection> rv1;
  }
}
