#ifndef Alignment_CommonAlignmentAlgorithm_TrackerAlignmentProducer_h
#define Alignment_CommonAlignmentAlgorithm_TrackerAlignmentProducer_h

//
// Package:    Alignment/CommonAlignmentAlgorithm
// Class:      AlignmentProducer
// 
//
// Description: calls alignment algorithms
//
//
// Original Author:  Frederic Ronga


#include <vector>

#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

class AlignmentProducer : public TrackProducerBase, public edm::EDProducer 
{

 public:

  /// Constructor
  explicit AlignmentProducer(const edm::ParameterSet& iConfig);

  /// Called at each event (if product is requested)
  virtual void produce(edm::Event&, const edm::EventSetup&);

  /// Called at beginning of job
  virtual void beginJob(EventSetup const&);

  /// Called at end of job
  virtual void endJob();

 private:

  TrackProducerAlgorithm theRefitterAlgo;

  AlignmentAlgorithmBase* theAlignmentAlgo;

  AlignableTracker* theAlignableTracker;

  // Helper method to only keep trajectories
  std::vector<Trajectory*> getTrajectories(AlgoProductCollection algoResults);

};

#endif
