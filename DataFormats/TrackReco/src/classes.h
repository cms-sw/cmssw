#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include <vector>

namespace {
  namespace {
    reco::TrackExtraCollection v3;
    edm::Wrapper<reco::TrackExtraCollection> c3;
    edm::Ref<reco::TrackExtraCollection> r3;
    edm::RefProd<reco::TrackExtraCollection> rp3;
    edm::RefVector<reco::TrackExtraCollection> rv3;

    reco::TrackCollection v1;
    edm::Wrapper<reco::TrackCollection> c1;
    reco::TrackRef r1;
    reco::TrackRefProd rp1;
    reco::TrackRefVector rv1;
    edm::Wrapper<reco::TrackRefVector> wv1;

    edm::helpers::Key< edm::RefProd < std::vector < reco::Track > > > rpt1;
    edm::AssociationMap<edm::OneToValue< std::vector<reco::Track>, double, unsigned int > > am1;
  }
}
