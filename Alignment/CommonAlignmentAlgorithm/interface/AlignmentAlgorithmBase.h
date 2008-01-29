#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentAlgorithmBase_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentAlgorithmBase_h

//
// Base class for the alignment algorithm
//
// Any algorithm should derive from this class
//

#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
// GF: I prefer the following instead, but needs fixed includes in CSA06AlignmentAlgorithm:
// namespace reco {
//   class Track;
// }

// Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

// Alignment
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
// GF: I prefer the following, but no time to check for possible fixes needed in other code...
// class AlignableTracker;
// class AlignableMuon;
// class AlignmentParameterStore;


class Trajectory;

class AlignmentAlgorithmBase
{

public:

  typedef std::pair<const Trajectory*, const reco::Track*> ConstTrajTrackPair; 
  typedef std::vector< ConstTrajTrackPair >  ConstTrajTrackPairCollection;
  
  /// Constructor
  AlignmentAlgorithmBase(const edm::ParameterSet& cfg);
  
  /// Destructor
  virtual ~AlignmentAlgorithmBase() {};

  /// Call at beginning of job (must be implemented in derived class)
  virtual void initialize( const edm::EventSetup& setup, 
                           AlignableTracker* tracker,
                           AlignableMuon* muon,
                           AlignmentParameterStore* store ) = 0;
   /// Call at start of loop
   /// Default implementation is dummy for non-iterative algorithms
   virtual void startNewLoop(void) {}

  /// Call at end of job (must be implemented in derived class)
  virtual void terminate(void) = 0;

  /// Run the algorithm on trajectories and tracks (must be implemented in derived class)
  virtual void run( const edm::EventSetup& setup,
                    const ConstTrajTrackPairCollection& tracks ) = 0;

};

#endif
