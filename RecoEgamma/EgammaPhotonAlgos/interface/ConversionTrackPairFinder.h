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
//
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
//
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
#include "DataFormats/EgammaTrackReco/interface/TrackSuperClusterAssociation.h"

class ConversionTrackPairFinder {

public:

  ConversionTrackPairFinder();


  ~ConversionTrackPairFinder();


  //   std::vector<std::vector<reco::TransientTrack> > run(std::vector<reco::TransientTrack> outIn,  
  //					      const edm::Handle<reco::TrackCollection>&  outInTrkHandle,
  //					      const edm::Handle<reco::TrackSuperClusterAssociationCollection>&  outInTrackSCAssH, 
  //					      std::vector<reco::TransientTrack> inOut,  
  //					      const edm::Handle<reco::TrackCollection>& inOutTrkHandle,
  //					      const edm::Handle<reco::TrackSuperClusterAssociationCollection>& inOutTrackSCAssH  );

  std::map<std::vector<reco::TransientTrack>, const reco::SuperCluster*>  run(std::vector<reco::TransientTrack> outIn,  
  					      const edm::Handle<reco::TrackCollection>&  outInTrkHandle,
  					      const edm::Handle<reco::TrackSuperClusterAssociationCollection>&  outInTrackSCAssH, 
  					      std::vector<reco::TransientTrack> inOut,  
  					      const edm::Handle<reco::TrackCollection>& inOutTrkHandle,
  					      const edm::Handle<reco::TrackSuperClusterAssociationCollection>& inOutTrackSCAssH  );


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


