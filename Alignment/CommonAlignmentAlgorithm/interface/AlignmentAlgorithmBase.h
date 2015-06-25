
#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentAlgorithmBase_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentAlgorithmBase_h

///
/// Base class for the alignment algorithm
///
/// Each alignment algorithm should derive from this class
///

/*** System includes ***/
#include <vector>
#include <utility>

/*** Core framework functionality ***/
#include "CondCore/DBCommon/interface/Time.h"
#include "FWCore/Framework/interface/Event.h"
namespace edm  { class EventSetup; class ParameterSet; }
namespace reco { class Track; class BeamSpot; }

/*** Alignment ***/
// These data formats cannot be forward declared since they are typedefs,
// hence, include the headers that define the typedef's (no need to include
// in dependencies in BuildFile):
// class TsosVectorCollection;
// class TkFittedLasBeamCollection;
// class AliClusterValueMap;
#include "Alignment/LaserAlignment/interface/TsosVectorCollection.h"
#include "DataFormats/Alignment/interface/TkFittedLasBeamCollectionFwd.h"
#include "DataFormats/Alignment/interface/AliClusterValueMapFwd.h"

class AlignableTracker;
class AlignableMuon;
class AlignableExtras;
class AlignmentParameterStore;

/*** Global typedefs part I (see EOF for part II) ***/
class Trajectory;
typedef std::pair<const Trajectory*, const reco::Track*> ConstTrajTrackPair;
typedef std::vector< ConstTrajTrackPair >                ConstTrajTrackPairs;

class IntegratedCalibrationBase;
typedef std::vector<IntegratedCalibrationBase*> Calibrations;

typedef cond::RealTimeType<cond::runnumber>::type RunNumber;
typedef std::pair<RunNumber,RunNumber>            RunRange;
typedef std::vector<RunRange>                     RunRanges;



class AlignmentAlgorithmBase {
  public:
    // TODO: DEPRECATED: For not breaking the interface, used in serveral files.
    //                   If possible use the global typedefs above.
    typedef std::pair<const Trajectory*, const reco::Track*> ConstTrajTrackPair;
    typedef ConstTrajTrackPairs ConstTrajTrackPairCollection;
    typedef cond::RealTimeType<cond::runnumber>::type RunNumber;
    typedef std::pair<RunNumber,RunNumber>            RunRange;
    typedef std::vector<RunRange>                     RunRanges;



    /// Define event information passed to algorithms
    class EventInfo {
      public:
        EventInfo(const edm::EventID&        theEventId,
                  const ConstTrajTrackPairs& theTrajTrackPairs,
                  const reco::BeamSpot&      theBeamSpot,
                  const AliClusterValueMap*  theClusterValueMap) :
          eventId_        (theEventId),
          trajTrackPairs_ (theTrajTrackPairs),
          beamSpot_       (theBeamSpot),
          clusterValueMap_(theClusterValueMap) {}

        const edm::EventID eventId()                const { return eventId_; }
        const ConstTrajTrackPairs& trajTrackPairs() const { return trajTrackPairs_; }
        const reco::BeamSpot& beamSpot()            const { return beamSpot_; }
        const AliClusterValueMap* clusterValueMap() const { return clusterValueMap_; } /// might be null!

      private:
        const edm::EventID         eventId_;
        const ConstTrajTrackPairs& trajTrackPairs_;
        const reco::BeamSpot&      beamSpot_;
        const AliClusterValueMap*  clusterValueMap_; /// might be null!
    };

    /// Define run information passed to algorithms (in endRun)
    class EndRunInfo {
      public:
        EndRunInfo(const edm::RunID&                theRunId,
                   const TkFittedLasBeamCollection* theTkLasBeams,
                   const TsosVectorCollection*      theTkLasBeamTsoses) :
          runId_(theRunId),
          tkLasBeams_(theTkLasBeams),
          tkLasBeamTsoses_(theTkLasBeamTsoses) {}

        const edm::RunID runId()                      const { return runId_; }
        const TkFittedLasBeamCollection* tkLasBeams() const { return tkLasBeams_; } /// might be null!
        const TsosVectorCollection* tkLasBeamTsoses() const { return tkLasBeamTsoses_; } /// might be null!

      private:
        const edm::RunID                 runId_;
        const TkFittedLasBeamCollection* tkLasBeams_; /// might be null!
        const TsosVectorCollection*      tkLasBeamTsoses_; /// might be null!
    };



    /// Constructor
    AlignmentAlgorithmBase(const edm::ParameterSet&) {}
    /// Destructor
    virtual ~AlignmentAlgorithmBase() {}

    /// Called at beginning of job (must be implemented in derived class)
    virtual void initialize(const edm::EventSetup&,
                            AlignableTracker*,
                            AlignableMuon*,
                            AlignableExtras*,
                            AlignmentParameterStore*) = 0;
    /// Called at end of job (must be implemented in derived class)
    virtual void terminate(const edm::EventSetup&) = 0;
    /// Called at end of job (must be implemented in derived class)
    virtual void terminate() {}

    /// Returns whether calibrations is supported by algorithm,
    /// default implementation returns false.
    virtual bool supportsCalibrations() { return false; }
    /// Pass integrated calibrations to algorithm, to be called after initialize()
    /// Calibrations' ownership is NOT passed to algorithm
    virtual void addCalibrations(const Calibrations&) {}

    // TODO: DEPRECATED: Actually, there are no iterative algorithms, use
    //                   initialze() and terminate()
    /// Called at start of loop, default implementation is dummy for
    /// non-iterative algorithms
    virtual void startNewLoop() {}
    /// Called at end of loop, default implementation is dummy for
    /// non-iterative algorithms
    virtual void endLoop() {}

    /// Called at begin of run
    virtual void beginRun(const edm::EventSetup&) {}
    /// Called at end of run - order of arguments like in EDProducer etc.
    virtual void endRun  (const EndRunInfo&, const edm::EventSetup&) {}

    /// Called at begin of luminosity block (no lumi block info passed yet)
    virtual void beginLuminosityBlock(const edm::EventSetup&) {}
    /// Called at end of luminosity block (no lumi block info passed yet)
    virtual void endLuminosityBlock  (const edm::EventSetup&) {}

    /// Returns whether algorithm proccesses events in current configuration
    virtual bool processesEvents() { return true; }
    /// Run the algorithm (must be implemented in derived class)
    virtual void run(const edm::EventSetup&, const EventInfo&) = 0;

    /// Called in order to pass parameters to alignables for a specific run
    /// range in case the algorithm supports run range dependent alignment.
    virtual bool setParametersForRunRange(const RunRange&) { return false; }
};

/*** Global typedefs part II ***/
typedef AlignmentAlgorithmBase::EventInfo  EventInfo;
typedef AlignmentAlgorithmBase::EndRunInfo EndRunInfo;

#endif
