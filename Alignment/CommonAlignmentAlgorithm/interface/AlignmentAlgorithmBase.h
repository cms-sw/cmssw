#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentAlgorithmBase_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentAlgorithmBase_h

//
// Base class for the alignment algorithm
//
// Any algorithm should derive from this class
//

#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"

// Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

// Alignment
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"

// Track refit
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"

class Trajectory;

class AlignmentAlgorithmBase : public TrackProducerBase
{

public:

  typedef std::pair<Trajectory*, reco::Track*> TrajTrackPair; 
  typedef std::vector< TrajTrackPair >  TrajTrackPairCollection;
  
  /// Constructor
  AlignmentAlgorithmBase(const edm::ParameterSet& cfg);
  
  /// Destructor
  virtual ~AlignmentAlgorithmBase() {};

  /// Dummy implementation 
  // Needed for inheritance of TrackProducerBase, but we don't produce anything
  virtual void produce(edm::Event&, const edm::EventSetup&) {};

  /// Call at beginning of job (must be implemented in derived class)
  virtual void initialize( const edm::EventSetup& setup, 
						   AlignableTracker* tracker, 
						   AlignmentParameterStore* store ) = 0;

  /// Call at end of job (must be implemented in derived class)
  virtual void terminate(void) = 0;

  /// Run the algorithm on trajectories and tracks (must be implemented in derived class)
  virtual void run( const edm::EventSetup& setup,
					const TrajTrackPairCollection& tracks ) = 0;

  /// Default implementation of track refit
  virtual 
  AlgoProductCollection refitTracks( const edm::Event& event, const edm::EventSetup& setup );

protected:

  TrackProducerAlgorithm theRefitterAlgo;

  // To bypass TrackerProducerBase::getFromEvt, which requires non-const Event
  std::string theSrc; 

  bool debug;

};

#endif
