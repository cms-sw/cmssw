#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/GsfTrack.h"
#include "DataFormats/PixelMatchTrackReco/interface/TrackSeedAssociation.h"
#include "DataFormats/PixelMatchTrackReco/interface/GsfTrackSeedAssociation.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include <vector>

namespace {
  namespace {
    reco::TrackSeedAssociationCollection v5;
    edm::Wrapper<reco::TrackSeedAssociationCollection> c5;
    reco::TrackSeedAssociation vv5;
    reco::TrackSeedAssociationRef r5;
    reco::TrackSeedAssociationRefProd rp5;
    reco::TrackSeedAssociationRefVector rv5;
    
    reco::GsfTrackSeedAssociationCollection v55;
    edm::Wrapper<reco::GsfTrackSeedAssociationCollection> c55;
    reco::GsfTrackSeedAssociation vv55;
    reco::GsfTrackSeedAssociationRef r55;
    reco::GsfTrackSeedAssociationRefProd rp55;
    reco::GsfTrackSeedAssociationRefVector rv55;
  }
}
