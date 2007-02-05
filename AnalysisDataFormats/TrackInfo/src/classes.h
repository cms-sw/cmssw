#include "DataFormats/Common/interface/Wrapper.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfoTrackAssociation.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include <vector>


namespace {
  namespace {
    reco::TrackInfoCollection v4;
    edm::Wrapper<reco::TrackInfoCollection> c4;
    reco::TrackInfoRef r1;
    reco::TrackInfoRefProd rp1;
    reco::TrackInfoRefVector rv1;
    edm::Wrapper<reco::TrackInfoRefVector> wv1;

    reco::TrackInfoTrackAssociationCollection v5;
    edm::Wrapper<reco::TrackInfoTrackAssociationCollection> c5;
    reco::TrackInfoTrackAssociation vv5;
    reco::TrackInfoTrackAssociationRef r5;
    reco::TrackInfoTrackAssociationRefProd rp5;
    reco::TrackInfoTrackAssociationRefVector rv5;
  }
}
