#ifndef Alignment_CommonAlignmentAlgorithm_TrackerAlignmentProducer_h
#define Alignment_CommonAlignmentAlgorithm_TrackerAlignmentProducer_h

/// \class AlignmentProducer
///
/// Package     : Alignment/CommonAlignmentProducer
/// Description : calls alignment algorithms
///
///  \author    : Frederic Ronga
///  Revision   : $Revision: 1.27 $
///  last update: $Date: 2012/06/13 16:23:30 $
///  by         : $Author: yana $

#include <memory>
#include <vector>

// Framework
#include "FWCore/Framework/interface/ESProducerLooper.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "DataFormats/Provenance/interface/RunID.h"

// Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

// Alignment
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h" 
#include <Geometry/Records/interface/MuonGeometryRecord.h> 
#include "Alignment/CommonAlignment/interface/AlignableExtras.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include <FWCore/Framework/interface/Frameworkfwd.h>
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentSurfaceDeformations.h"

// for watcher
#include "CondFormats/AlignmentRecord/interface/TrackerSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurveyErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTSurveyErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCSurveyErrorExtendedRcd.h"


class Alignments;
class IntegratedCalibrationBase;
struct SurveyErrors;
namespace edm {
  class Run;
  class LuminosityBlock;
}

class AlignmentProducer : public edm::ESProducerLooper
{

 public:
  using Alignables = align::Alignables;
  typedef std::pair<const Trajectory*, const reco::Track*> ConstTrajTrackPair; 
  typedef std::vector<ConstTrajTrackPair>  ConstTrajTrackPairCollection;

  using RunNumber = align::RunNumber;
  using RunRange = align::RunRange;
  using RunRanges = align::RunRanges;

  /// Constructor
  AlignmentProducer( const edm::ParameterSet& iConfig );
  
  /// Destructor
  ~AlignmentProducer();

  /// Produce the tracker geometry
  virtual std::shared_ptr<TrackerGeometry> produceTracker( const TrackerDigiGeometryRecord& iRecord );
  /// Produce the muon DT geometry
  virtual std::shared_ptr<DTGeometry>      produceDT( const MuonGeometryRecord& iRecord );
  /// Produce the muon CSC geometry
  virtual std::shared_ptr<CSCGeometry>     produceCSC( const MuonGeometryRecord& iRecord );

  /// Called at beginning of job
  virtual void beginOfJob(const edm::EventSetup&);

  /// Called at end of job
  virtual void endOfJob();

  /// Called at beginning of loop
  virtual void startingNewLoop( unsigned int iLoop );

  /// Called at end of loop
  virtual Status endOfLoop( const edm::EventSetup&, unsigned int iLoop );

  /// Called at run start and calling algorithms beginRun
  virtual void beginRun(const edm::Run &run, const edm::EventSetup &setup);
  /// Called at run end - currently reading TkFittedLasBeam if an InpuTag is given for that
  virtual void endRun(const edm::Run &run, const edm::EventSetup &setup);

  /// Called at lumi block start, calling algorithm's beginLuminosityBlock
  virtual void beginLuminosityBlock(const edm::LuminosityBlock &lumiBlock,
				    const edm::EventSetup &setup);
  /// Called at lumi block end, calling algorithm's endLuminosityBlock
  virtual void endLuminosityBlock(const edm::LuminosityBlock &lumiBlock,
				  const edm::EventSetup &setup);
  /// Called at each event 
  virtual Status duringLoop(const edm::Event &event, const edm::EventSetup &setup);

 private:

  // private member functions

  /// Creates the choosen alignment algorithm (specified in config-file)
  void createAlignmentAlgorithm(const edm::ParameterSet&);

  /// Checks if one of the EventSetup-Records has changed
  bool setupChanged(const edm::EventSetup&);

  /// Creates Geometry and Alignables of the Tracker and initializes the
  /// AlignmentAlgorithm @theAlignmentAlgo
  void initAlignmentAlgorithm(const edm::EventSetup&, bool update = false);

  /// Applies Alignments from Database (GlobalPositionRcd) to Geometry
  /// @theTracker
  void applyAlignmentsToDB(const edm::EventSetup&);

  /// Creates Alignables @theAlignableTracker from the previously loaded
  /// Geometry @theTracker
  void createAlignables(const TrackerTopology*, bool update = false);

  /// Creates the @theAlignmentParameterStore, which manages all Alignables
  void buildParameterStore();

  /// Applies misalignment scenario to @theAlignableTracker
  void applyMisalignment();

  /// Apply random shifts and rotations to selected alignables, according to configuration
  void simpleMisalignment(const Alignables &alivec, const std::string &selection,
                          float shift, float rot, bool local);

  /// Create tracker and muon geometries
  void createGeometries(const edm::EventSetup&, const TrackerTopology*);

  /// Applies Alignments, AlignmentErrors and SurfaceDeformations to
  /// @theTracker
  void applyAlignmentsToGeometry();

  /// Apply DB constants belonging to (Err)Rcd to geometry,
  /// taking into account 'globalPosition' correction.
  template<class G, class Rcd, class ErrRcd>
    void applyDB(G *geometry, const edm::EventSetup &iSetup,
		 const AlignTransform &globalPosition) const;
  /// Apply DB constants for surface deformations
  template<class G, class DeformationRcd>
    void applyDB(G *geometry, const edm::EventSetup &iSetup) const;

  // write alignments and alignment errors for all sub detectors and
  // the given run number
  void writeForRunRange(cond::Time_t time);

  /// Write alignment and/or errors to DB for record names
  /// (removes *globalCoordinates before writing if non-null...).
  /// Takes over ownership of alignments and alignmentErrrors.
  void writeDB(Alignments *alignments, const std::string &alignRcd,
	       AlignmentErrorsExtended *alignmentErrors, const std::string &errRcd,
	       const AlignTransform *globalCoordinates,
	       cond::Time_t time) const;
  /// Write surface deformations (bows & kinks) to DB for given record name
  /// Takes over ownership of alignmentsurfaceDeformations.
  void writeDB(AlignmentSurfaceDeformations *alignmentSurfaceDeformations,
	       const std::string &surfaceDeformationRcd,
	       cond::Time_t time) const;

  /// Add survey info to an alignable
  void addSurveyInfo_(Alignable*);
	
  /// read in survey records
  void readInSurveyRcds( const edm::EventSetup& );


  // private data members

  unsigned int        theSurveyIndex;
  const Alignments*   theSurveyValues;
  const SurveyErrors* theSurveyErrors;

  AlignmentAlgorithmBase* theAlignmentAlgo;
  AlignmentParameterStore* theAlignmentParameterStore;
  std::vector<AlignmentMonitorBase*> theMonitors;
  std::vector<IntegratedCalibrationBase*> theCalibrations;

  AlignableExtras* theAlignableExtras;
  AlignableTracker* theAlignableTracker;
  AlignableMuon* theAlignableMuon;

  std::shared_ptr<TrackerGeometry> theTracker;
  std::shared_ptr<DTGeometry> theMuonDT;
  std::shared_ptr<CSCGeometry> theMuonCSC;
  /// GlobalPositions that might be read from DB, NULL otherwise
  const Alignments *globalPositions_;

  const RunRanges uniqueRunRanges_;
  int nevent_;
  edm::ParameterSet theParameterSet;

  // steering parameters

  const unsigned int theMaxLoops;     // Number of loops to loop

  const int stNFixAlignables_;
  const double stRandomShift_,stRandomRotation_;
  const bool applyDbAlignment_,checkDbAlignmentValidity_;
  const bool doMisalignmentScenario_;
  const bool saveToDB_, saveApeToDB_,saveDeformationsToDB_;
  const bool doTracker_,doMuon_,useExtras_;
  const bool useSurvey_; // true to read survey info from DB
  const bool enableAlignableUpdates_;

  // event input tags
  const edm::InputTag tjTkAssociationMapTag_; // map with tracks/trajectories
  const edm::InputTag beamSpotTag_;           // beam spot
  const edm::InputTag tkLasBeamTag_;          // LAS beams in edm::Run (ignore if empty)
  const edm::InputTag clusterValueMapTag_;              // ValueMap containing associtaion cluster - flag

  // ESWatcher

  edm::ESWatcher<IdealGeometryRecord> watchIdealGeometryRcd;
  edm::ESWatcher<GlobalPositionRcd> watchGlobalPositionRcd;

  edm::ESWatcher<TrackerAlignmentRcd> watchTrackerAlRcd;
  edm::ESWatcher<TrackerAlignmentErrorExtendedRcd> watchTrackerAlErrorExtRcd;
  edm::ESWatcher<TrackerSurfaceDeformationRcd> watchTrackerSurDeRcd;

  edm::ESWatcher<DTAlignmentRcd> watchDTAlRcd;
  edm::ESWatcher<DTAlignmentErrorExtendedRcd> watchDTAlErrExtRcd;
  edm::ESWatcher<CSCAlignmentRcd> watchCSCAlRcd;
  edm::ESWatcher<CSCAlignmentErrorExtendedRcd> watchCSCAlErrExtRcd;

  edm::ESWatcher<TrackerSurveyRcd> watchTkSurveyRcd_;
  edm::ESWatcher<TrackerSurveyErrorExtendedRcd> watchTkSurveyErrRcd_;
  edm::ESWatcher<DTSurveyRcd> watchDTSurveyRcd_;
  edm::ESWatcher<DTSurveyErrorExtendedRcd> watchDTSurveyErrRcd_;
  edm::ESWatcher<CSCSurveyRcd> watchCSCSurveyRcd_;
  edm::ESWatcher<CSCSurveyErrorExtendedRcd> watchCSCSurveyErrRcd_;	

};

#endif
