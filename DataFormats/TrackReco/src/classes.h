#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/TrackReco/interface/HelixParameters.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/RecHit.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include <vector>

namespace {
  namespace {
    reco::RecHitCollection v2;
    edm::Wrapper<reco::RecHitCollection> c2;
    edm::Ref<reco::RecHitCollection> r2;
    edm::RefProd<reco::RecHitCollection> rp2;
    edm::RefVector<reco::RecHitCollection> rv2;

    reco::TrackExtraCollection v3;
    edm::Wrapper<reco::TrackExtraCollection> c3;
    edm::Ref<reco::TrackExtraCollection> r3;
    edm::RefProd<reco::TrackExtraCollection> rp3;
    edm::RefVector<reco::TrackExtraCollection> rv3;

    reco::TrackCollection v1;
    edm::Wrapper<reco::TrackCollection> c1;
    edm::Ref<reco::TrackCollection> r1;
    edm::RefProd<reco::TrackCollection> rp1;
    edm::RefVector<reco::TrackCollection> rv1;
  }
}
