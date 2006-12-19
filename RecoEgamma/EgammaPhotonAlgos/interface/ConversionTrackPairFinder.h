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
//
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"


class ConversionTrackPairFinder {

public:

  ConversionTrackPairFinder();


  ~ConversionTrackPairFinder();


  // void run(edm::Event& evt, const edm::Handle<reco::TrackCollection>& outIn,  const edm::Handle<reco::TrackCollection>& inOut  );
 std::vector<std::vector<reco::Track> > run(const edm::Handle<reco::TrackCollection>& outIn,  const edm::Handle<reco::TrackCollection>& inOut  );


 std::vector<std::vector<reco::TransientTrack> > run(std::vector<reco::TransientTrack> outIn,  std::vector<reco::TransientTrack> inOut  );


 private:

class ByNumOfHits {
 public:
  bool operator()(reco::Track a, reco::Track  b) {
    if (a.numberOfValidHits()  == b.numberOfValidHits()  ) {
      return a.normalizedChi2() < b.normalizedChi2();
    } else {
      return a.numberOfValidHits()  > b.numberOfValidHits() ;
    }
  }
};




};

#endif // ConversionTrackPairFinder_H


