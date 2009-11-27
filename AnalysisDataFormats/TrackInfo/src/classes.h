#include "DataFormats/Common/interface/Wrapper.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfoTrackAssociation.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "AnalysisDataFormats/TrackInfo/interface/TPtoRecoTrack.h"
#include "AnalysisDataFormats/TrackInfo/interface/TPtoRecoTrackCollection.h"
#include "AnalysisDataFormats/TrackInfo/interface/RecoTracktoTP.h"
#include "AnalysisDataFormats/TrackInfo/interface/RecoTracktoTPCollection.h"
#include <vector>


namespace {
  struct dictionary {

    reco::TrackingStateInfo                 tsi;
    reco::TrackingRecHitInfo                tri;
    std::pair<reco::StateType, reco::TrackingStateInfo> ptsi;
    std::map<reco::StateType, reco::TrackingStateInfo>  mtsi;

    std::pair<TrackingRecHitRef, reco::TrackingRecHitInfo>                  ptri;
    std::map<TrackingRecHitRef, reco::TrackingRecHitInfo>                   mtri;
    
    
    reco::TrackInfo                         ti;
    reco::TrackInfoCollection               vti;
    edm::Wrapper<reco::TrackInfoCollection> cti;
    reco::TrackInfoRef                      rti;
    reco::TrackInfoRefProd                  rpti;
    reco::TrackInfoRefVector                rvti;
    edm::Wrapper<reco::TrackInfoRefVector>  wvti;

    reco::TrackInfoTrackAssociationCollection v5;
    edm::Wrapper<reco::TrackInfoTrackAssociationCollection> c5;
    reco::TrackInfoTrackAssociation vv5;
    reco::TrackInfoTrackAssociationRef r5;
    reco::TrackInfoTrackAssociationRefProd rp5;
    reco::TrackInfoTrackAssociationRefVector rv5;

    edm::Wrapper<TPtoRecoTrack> TPtoRecoTrackWrapper ;
    edm::Wrapper<TPtoRecoTrackCollection> TPtoRecoTrackCollectionWrapper;
    edm::Wrapper<RecoTracktoTP> RecoTracktoTPWrapper ;
    edm::Wrapper<RecoTracktoTPCollection> RecoTracktoTPCollectionWrapper;

  };
}
