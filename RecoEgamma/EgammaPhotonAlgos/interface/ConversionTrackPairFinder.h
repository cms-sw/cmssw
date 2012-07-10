#ifndef ConversionTrackPairFinder_H
#define ConversionTrackPairFinder_H

/** \class ConversionTrackPairFinder
 *
 *
 * \author N. Marinelli - Univ. of Notre Dame
 *
 * \version   
 *
 ************************************************************/


#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"

//
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"

//
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
#include "DataFormats/EgammaTrackReco/interface/TrackCaloClusterAssociation.h"

class ConversionTrackPairFinder {

public:

  ConversionTrackPairFinder();


  ~ConversionTrackPairFinder();



  std::map<std::vector<reco::TransientTrack>, reco::CaloClusterPtr>  run(std::vector<reco::TransientTrack> outIn,  
  					      const edm::Handle<reco::TrackCollection>&  outInTrkHandle,
  					      const edm::Handle<reco::TrackCaloClusterPtrAssociation>&  outInTrackSCAssH, 
  					      std::vector<reco::TransientTrack> inOut,  
  					      const edm::Handle<reco::TrackCollection>& inOutTrkHandle,
  					      const edm::Handle<reco::TrackCaloClusterPtrAssociation>& inOutTrackSCAssH  );




 private:

class ByNumOfHits {
 public:
  bool operator()(reco::TransientTrack a, reco::TransientTrack  b) {
    if (a.numberOfValidHits()  == b.numberOfValidHits()  ) {
      return a.normalizedChi2() < b.normalizedChi2();
    } else {
      return a.numberOfValidHits()  > b.numberOfValidHits() ;
    }
  }
};




};

#endif // ConversionTrackPairFinder_H


