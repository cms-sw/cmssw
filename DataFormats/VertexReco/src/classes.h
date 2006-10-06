#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include <vector>
#include <map>

namespace {
  namespace {
    reco::Vertex rv1;
    std::vector<reco::Vertex> v1;
    edm::Wrapper<std::vector<reco::Vertex> > wc1;
    edm::Ref<std::vector<reco::Vertex> > r1;
    edm::RefProd<std::vector<reco::Vertex> > rp1;
    edm::RefVector<std::vector<reco::Vertex> > rvv1;
    edm::helpers::Key< edm::RefProd < std::vector < reco::Track > > > rpt1;
    reco::TrackRef tr1; 
    reco::TrackRefVector trv1;
    // edm::AssociationMap<edm::OneToValue< std::vector<reco::Track>, double, unsigned int > > am1;
  }
}
