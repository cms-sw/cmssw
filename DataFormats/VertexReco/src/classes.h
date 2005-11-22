#include "FWCore/EDProduct/interface/Wrapper.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <vector>

namespace {
  namespace {
    reco::Error<3> e31;
    reco::Vector<3> v31;
    
    std::vector<reco::Track> t1;
    edm::Ref<std::vector<reco::Track> > ref1;
    edm::RefVector<std::vector<reco::Track> > refv1;
    std::vector<reco::Vertex> v1;
    edm::Wrapper<std::vector<reco::Vertex> > wc1;
  }
}
