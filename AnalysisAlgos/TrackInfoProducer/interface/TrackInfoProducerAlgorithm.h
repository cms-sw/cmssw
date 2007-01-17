#ifndef TrackInfoProducerAlgorithm_h
#define TrackInfoProducerAlgorithm_h

//
// Package:    RecoTracker/TrackProducer
// Class:      TrackInfoProducerAlgorithm
// 
//
// Original $Author:  Chiara Genta
//          $Created:  
// $Id: 
//
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

class TrackInfoProducerAlgorithm {
  
 public:
typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;

  TrackInfoProducerAlgorithm(const edm::ParameterSet& conf) : 
    conf_(conf)    { }

  ~TrackInfoProducerAlgorithm() {}
  
  void run(const std::vector<Trajectory> * inputcoll, edm::Handle<TrackingRecHitCollection> *rechits, reco::TrackInfoCollection &outputFwdcoll,
	   reco::TrackInfoCollection & outputBwdColl,reco::TrackInfoCollection &outputUpdatedColl, reco::TrackInfoCollection &outputCombinedColl);

 private:
  edm::ParameterSet conf_;
};

#endif
