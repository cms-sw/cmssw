#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentAlgorithmBase_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentAlgorithmBase_h

//
// Base class for the alignment algorithm
//
// Any algorithm should derive from this class
//

#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

class Trajectory;
class EventSetup;

class AlignmentAlgorithmBase
{

public:

 typedef std::pair<Trajectory*, reco::Track*> TrajTrackPair; 
 typedef std::vector< TrajTrackPair >  TrajTrackPairCollection;
  
  /// Constructor
  AlignmentAlgorithmBase() {}
  AlignmentAlgorithmBase(const edm::ParameterSet& cfg, AlignableTracker* tracker) {};

  /// Destructor
  virtual ~AlignmentAlgorithmBase() {}

  /// Call at beginning of job
  virtual void initialize( const EventSetup& setup ) {}

  /// Call at end of job
  virtual void terminate(void) {}

  /// Run the algorithm on trajectories and tracks
  //virtual void run(const std::vector<Trajectory*>& trajectories ) {}
  virtual void run(const TrajTrackPairCollection& tracks ) {}

};



#endif
