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
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

class TrackInfoProducerAlgorithm {
  
 public:
typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;

  TrackInfoProducerAlgorithm(const edm::ParameterSet& conf) : 
    conf_(conf)    { }

  ~TrackInfoProducerAlgorithm() {}
  
  void run(std::vector<Trajectory>::const_iterator input, reco::TrackRef track, 
	   reco::TrackInfo &outputFwd,
	   reco::TrackInfo &outputBwd,
	   reco::TrackInfo &outputUpdated, 
	   reco::TrackInfo &outputCombined,
	   const TrackerGeometry * tracker);
    
 private:
  edm::ParameterSet conf_;
};

#endif
