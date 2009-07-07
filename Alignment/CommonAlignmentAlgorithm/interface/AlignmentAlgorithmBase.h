#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentAlgorithmBase_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentAlgorithmBase_h

//
// Base class for the alignment algorithm
//
// Any algorithm should derive from this class
//

#include <vector>
#include <utility>

class AlignableTracker;
class AlignableMuon;
class AlignmentParameterStore;
class Trajectory;

namespace edm { class EventID; class EventSetup; class ParameterSet; }
namespace reco { class Track; class BeamSpot; }

class AlignmentAlgorithmBase
{

public:

  typedef std::pair<const Trajectory*, const reco::Track*> ConstTrajTrackPair; 
  typedef std::vector< ConstTrajTrackPair >  ConstTrajTrackPairCollection;

  /// define event information passed to algorithms
  struct EventInfo {
    EventInfo(const edm::EventID &eventId, 
	      const ConstTrajTrackPairCollection &trajTrackPairs,
	      const reco::BeamSpot &beamSpot) 
      : eventId_(eventId), trajTrackPairs_(trajTrackPairs), beamSpot_(beamSpot) {}

    const edm::EventID                 &eventId_;
    const ConstTrajTrackPairCollection &trajTrackPairs_;
    const reco::BeamSpot               &beamSpot_;
  };
  
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
  virtual void startNewLoop() {}

  /// Call at end of job (must be implemented in derived class)
  virtual void terminate() = 0;

  /// Run the algorithm (must be implemented in derived class)
  virtual void run( const edm::EventSetup &setup, const EventInfo &eventInfo) = 0;

};

#endif
