
#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentAlgorithmBase_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentAlgorithmBase_h

/**
 * @package   Alignment/CommonAlignmentAlgorithm
 * @file      AlignmentAlgorithmBase.h
 *
 * @author    ???
 *
 * Last update:
 * @author    Max Stark (max.stark@cern.ch)
 * @date      2015/07/16
 *
 * @brief     Interface/Base class for alignment algorithms, each alignment
 *            algorithm has to be derived from this class
 */

#include <vector>
#include <utility>
#include <memory>

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
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/LaserAlignment/interface/TsosVectorCollection.h"
#include "DataFormats/Alignment/interface/TkFittedLasBeamCollectionFwd.h"
#include "DataFormats/Alignment/interface/AliClusterValueMapFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

namespace edm {
  class EventSetup;
  class ParameterSet;
}  // namespace edm
namespace reco {
  class Track;
  class BeamSpot;
}  // namespace reco

/*** Global typedefs part I (see EOF for part II) ***/
typedef std::pair<const Trajectory *, const reco::Track *> ConstTrajTrackPair;
typedef std::vector<ConstTrajTrackPair> ConstTrajTrackPairs;

typedef std::vector<IntegratedCalibrationBase *> Calibrations;
typedef std::vector<std::unique_ptr<IntegratedCalibrationBase>> CalibrationsOwner;

typedef std::vector<IntegratedCalibrationBase *> Calibrations;

class AlignmentAlgorithmBase {
public:
  // TODO: DEPRECATED: For not breaking the interface, used in serveral files.
  //                   If possible use the global typedefs above.
  // With global typedefs one does not have to typedef again like
  // 'typedef AlignmentAlgorithmBase::ConstTrajTrackPair ConstTrajTrackPair;'
  // in other files.
  typedef std::pair<const Trajectory *, const reco::Track *> ConstTrajTrackPair;
  typedef std::vector<ConstTrajTrackPair> ConstTrajTrackPairCollection;
  using RunNumber = align::RunNumber;
  using RunRange = align::RunRange;

  /// define event information passed to algorithms
  class EventInfo {
  public:
    EventInfo(const edm::EventID &theEventId,
              const ConstTrajTrackPairCollection &theTrajTrackPairs,
              const reco::BeamSpot &theBeamSpot,
              const AliClusterValueMap *theClusterValueMap)
        : eventId_(theEventId),
          trajTrackPairs_(theTrajTrackPairs),
          beamSpot_(theBeamSpot),
          clusterValueMap_(theClusterValueMap) {}

    const edm::EventID eventId() const { return eventId_; }
    const ConstTrajTrackPairCollection &trajTrackPairs() const { return trajTrackPairs_; }
    const reco::BeamSpot &beamSpot() const { return beamSpot_; }
    const AliClusterValueMap *clusterValueMap() const { return clusterValueMap_; }  ///might be null!

  private:
    const edm::EventID eventId_;
    const ConstTrajTrackPairCollection &trajTrackPairs_;
    const reco::BeamSpot &beamSpot_;
    const AliClusterValueMap *clusterValueMap_;  ///might be null!
  };

  /// define run information passed to algorithms (in endRun)
  class EndRunInfo {
  public:
    EndRunInfo(const edm::RunID &theRunId,
               const TkFittedLasBeamCollection *theTkLasBeams,
               const TsosVectorCollection *theTkLasBeamTsoses)
        : runId_(theRunId), tkLasBeams_(theTkLasBeams), tkLasBeamTsoses_(theTkLasBeamTsoses) {}

    const edm::RunID runId() const { return runId_; }
    const TkFittedLasBeamCollection *tkLasBeams() const { return tkLasBeams_; }       /// might be null!
    const TsosVectorCollection *tkLasBeamTsoses() const { return tkLasBeamTsoses_; }  /// might be null!

  private:
    const edm::RunID runId_;
    const TkFittedLasBeamCollection *tkLasBeams_;  /// might be null!
    const TsosVectorCollection *tkLasBeamTsoses_;  /// might be null!
  };

  /// Constructor
  AlignmentAlgorithmBase(const edm::ParameterSet &, const edm::ConsumesCollector &){};

  /// Destructor
  virtual ~AlignmentAlgorithmBase(){};

  /// Call at beginning of job (must be implemented in derived class)
  virtual void initialize(const edm::EventSetup &setup,
                          AlignableTracker *tracker,
                          AlignableMuon *muon,
                          AlignableExtras *extras,
                          AlignmentParameterStore *store) = 0;

  /// Returns whether calibrations is supported by algorithm,
  /// default implementation returns false.
  virtual bool supportsCalibrations() { return false; }
  /// Pass integrated calibrations to algorithm, to be called after initialize()
  /// Calibrations' ownership is NOT passed to algorithm
  virtual bool addCalibrations(const Calibrations &) { return false; }
  // Overloading for the owning vector
  bool addCalibrations(const CalibrationsOwner &cals) {
    Calibrations tmp;
    tmp.reserve(cals.size());
    for (const auto &ptr : cals) {
      tmp.push_back(ptr.get());
    }
    return addCalibrations(tmp);
  }

  /// Returns whether algorithm proccesses events in current configuration
  virtual bool processesEvents() { return true; }

  /// Returns whether algorithm produced results to be stored
  virtual bool storeAlignments() { return true; }

  // TODO: DEPRECATED: Actually, there are no iterative algorithms, use
  //                   initialze() and terminate()
  /// Called at start of loop, default implementation is dummy for
  /// non-iterative algorithms
  virtual void startNewLoop() {}

  /// Call at end of each loop (must be implemented in derived class)
  virtual void terminate(const edm::EventSetup &iSetup) = 0;
  /// Called at end of job (must be implemented in derived class)
  virtual void terminate() {}

  /// Run the algorithm (must be implemented in derived class)
  virtual void run(const edm::EventSetup &setup, const EventInfo &eventInfo) = 0;

  /// called at begin of run
  virtual void beginRun(const edm::Run &, const edm::EventSetup &, bool changed){};

  /// called at end of run - order of arguments like in EDProducer etc.
  virtual void endRun(const EndRunInfo &runInfo, const edm::EventSetup &setup){};

  /// called at begin of luminosity block (no lumi block info passed yet)
  virtual void beginLuminosityBlock(const edm::EventSetup &setup){};

  /// called at end of luminosity block (no lumi block info passed yet)
  virtual void endLuminosityBlock(const edm::EventSetup &setup){};

  /// called in order to pass parameters to alignables for a specific run
  /// range in case the algorithm supports run range dependent alignment.
  virtual bool setParametersForRunRange(const RunRange &rr) { return false; };
};

/*** Global typedefs part II ***/
typedef AlignmentAlgorithmBase::EventInfo EventInfo;
typedef AlignmentAlgorithmBase::EndRunInfo EndRunInfo;

#endif
