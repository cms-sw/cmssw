
#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentAlgorithmBase_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentAlgorithmBase_h

///
/// Base class for the alignment algorithm
///
/// Any algorithm should derive from this class
///

#include <vector>
#include <utility>

class AlignableTracker;
class AlignableMuon;
class AlignableExtras;
class AlignmentParameterStore;
class IntegratedCalibrationBase;
class Trajectory;
// These data formats cannot be forward declared since they are typedef's,
// so include the headers that define the typedef's
// (no need to include in dependencies in BuildFile):
// class TsosVectorCollection;
// class TkFittedLasBeamCollection;
// class AliClusterValueMap;
#include "CondCore/DBCommon/interface/Time.h"
#include "Alignment/LaserAlignment/interface/TsosVectorCollection.h"
#include "DataFormats/Alignment/interface/TkFittedLasBeamCollectionFwd.h"
#include "DataFormats/Alignment/interface/AliClusterValueMapFwd.h"

namespace edm { class EventID; class RunID; class EventSetup; class ParameterSet; }
namespace reco { class Track; class BeamSpot; }

class AlignmentAlgorithmBase
{

public:

  typedef std::pair<const Trajectory*, const reco::Track*> ConstTrajTrackPair; 
  typedef std::vector< ConstTrajTrackPair >  ConstTrajTrackPairCollection;
  typedef cond::RealTimeType<cond::runnumber>::type RunNumber;
  typedef std::pair<RunNumber,RunNumber> RunRange;

  /// define event information passed to algorithms
  struct EventInfo {
    EventInfo(const edm::EventID &eventId, 
	      const ConstTrajTrackPairCollection &trajTrackPairs,
	      const reco::BeamSpot &beamSpot,
	      const AliClusterValueMap *clusterValueMap) 
      : eventId_(eventId), trajTrackPairs_(trajTrackPairs), beamSpot_(beamSpot), clusterValueMap_(clusterValueMap) {}

    const edm::EventID                 &eventId_;
    const ConstTrajTrackPairCollection &trajTrackPairs_;
    const reco::BeamSpot               &beamSpot_;
    const AliClusterValueMap           *clusterValueMap_;///might be null!
  };
  
  /// define run information passed to algorithms (in endRun)
  struct EndRunInfo {
    EndRunInfo(const edm::RunID &runId, const TkFittedLasBeamCollection *tkLasBeams,
	       const TsosVectorCollection *tkLasBeamTsoses)
      : runId_(runId), tkLasBeams_(tkLasBeams), tkLasBeamTsoses_(tkLasBeamTsoses) {}
    const edm::RunID &runId_;
    const TkFittedLasBeamCollection *tkLasBeams_; /// might be null!
    const TsosVectorCollection *tkLasBeamTsoses_; /// might be null!
  };
  
  /// Constructor
  AlignmentAlgorithmBase(const edm::ParameterSet& cfg);
  
  /// Destructor
  virtual ~AlignmentAlgorithmBase() {};

  /// Call at beginning of job (must be implemented in derived class)
  virtual void initialize( const edm::EventSetup& setup, 
                           AlignableTracker* tracker,
                           AlignableMuon* muon,
                           AlignableExtras* extras,
                           AlignmentParameterStore* store ) = 0;
  /// Pass integrated calibrations to algorithm, to be called after initialize(..).
  /// (Calibrations' ownership is NOT passed to algorithm.)
  /// Return whether feature is supported by algorithm, 
  /// default implementation returns false.
  virtual bool addCalibrations(const std::vector<IntegratedCalibrationBase*> &iCals){return false;}

   /// Call at start of loop
   /// Default implementation is dummy for non-iterative algorithms
  virtual void startNewLoop() {}

  /// Call at end of each loop (must be implemented in derived class)
  virtual void terminate(const edm::EventSetup& iSetup) = 0;

  /// Run the algorithm (must be implemented in derived class)
  virtual void run( const edm::EventSetup &setup, const EventInfo &eventInfo) = 0;

  /// called at begin of run
  virtual void beginRun(const edm::EventSetup &setup) {};

  /// called at end of run - order of arguments like in EDProducer etc.
  virtual void endRun(const EndRunInfo &runInfo, const edm::EventSetup &setup) {};

  /// called at begin of luminosity block (no lumi block info passed yet)
  virtual void beginLuminosityBlock(const edm::EventSetup &setup) {};

  /// called at end of luminosity block (no lumi block info passed yet)
  virtual void endLuminosityBlock(const edm::EventSetup &setup) {};

  /// called in order to pass parameters to alignables for a specific run
  /// range in case the algorithm supports run range dependent alignment.
  virtual bool setParametersForRunRange(const RunRange& rr) { return false; };
};

#endif
