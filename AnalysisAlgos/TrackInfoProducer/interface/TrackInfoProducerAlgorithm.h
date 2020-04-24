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
typedef TrackingRecHit::ConstRecHitPointer ConstRecHitPointer;

  TrackInfoProducerAlgorithm(const edm::ParameterSet& conf) : 
    conf_(conf),
    forwardPredictedStateTag_(conf.getParameter<std::string>( "forwardPredictedState" )),
    backwardPredictedStateTag_(conf.getParameter<std::string>( "backwardPredictedState" )),
    updatedStateTag_(conf.getParameter<std::string>( "updatedState" )),
    combinedStateTag_(conf.getParameter<std::string>( "combinedState" ))    { }

  ~TrackInfoProducerAlgorithm() {}
  
  void run(const edm::Ref<std::vector<Trajectory> > traj_iterator, reco::TrackRef track, 
	   reco::TrackInfo &output,
	   const TrackerGeometry * tracker);
  LocalPoint project(const GeomDet *det,const GeomDet* projdet,LocalPoint position,LocalVector trackdirection)const;
 private:
  edm::ParameterSet conf_;
  std::string forwardPredictedStateTag_, backwardPredictedStateTag_, updatedStateTag_, combinedStateTag_;
};

#endif
