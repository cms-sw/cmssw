#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include <vector>

namespace {
  namespace {
    reco::GsfTrackExtraCollection v4;
    edm::Wrapper<reco::GsfTrackExtraCollection> c4;
    edm::Ref<reco::GsfTrackExtraCollection> r4;
    edm::RefProd<reco::GsfTrackExtraCollection> rp4;
    edm::RefVector<reco::GsfTrackExtraCollection> rv4;

    reco::GsfTrackCollection v2;
    edm::Wrapper<reco::GsfTrackCollection> c2;
    edm::Ref<reco::GsfTrackCollection> r2;
    edm::RefProd<reco::GsfTrackCollection> rp2;
    edm::RefVector<reco::GsfTrackCollection> rv2;

    edm::helpers::Key< edm::RefProd < std::vector < reco::GsfTrack > > > rpt11;
    edm::AssociationMap<edm::OneToValue< std::vector<reco::GsfTrack>, double, unsigned int > > am11;
  }
}
