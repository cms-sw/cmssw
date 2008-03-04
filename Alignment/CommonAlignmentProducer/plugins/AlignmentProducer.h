#ifndef Alignment_CommonAlignmentAlgorithm_TrackerAlignmentProducer_h
#define Alignment_CommonAlignmentAlgorithm_TrackerAlignmentProducer_h

/// \class AlignmentProducer
///
/// Package     : Alignment/CommonAlignmentProducer
/// Description : calls alignment algorithms
///
///  \author    : Frederic Ronga
///  Revision   : $Revision: 1.5 $
///  last update: $Date: 2007/07/03 18:36:13 $
///  by         : $Author: cklae $

#include <vector>

// Framework
#include "FWCore/Framework/interface/LooperFactory.h"
#include "FWCore/Framework/interface/ESProducerLooper.h"
#include "FWCore/Framework/interface/ESHandle.h"

// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

// Alignment
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterBuilder.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"

class Alignments;
class SurveyErrors;

class AlignmentProducer : public edm::ESProducerLooper
{

 public:
  typedef std::vector<Alignable*> Alignables;
  typedef std::pair<const Trajectory*, const reco::Track*> ConstTrajTrackPair; 
  typedef std::vector<ConstTrajTrackPair>  ConstTrajTrackPairCollection;
  
  /// Constructor
  AlignmentProducer( const edm::ParameterSet& iConfig );
  
  /// Destructor
  ~AlignmentProducer();

  /// Produce the tracker geometry
  virtual boost::shared_ptr<TrackerGeometry> produceTracker( const TrackerDigiGeometryRecord& iRecord );
  /// Produce the muon DT geometry
  virtual boost::shared_ptr<DTGeometry>      produceDT( const MuonGeometryRecord& iRecord );
  /// Produce the muon CSC geometry
  virtual boost::shared_ptr<CSCGeometry>     produceCSC( const MuonGeometryRecord& iRecord );

  /// Called at beginning of job
  virtual void beginOfJob(const edm::EventSetup&);

  /// Called at end of job
  virtual void endOfJob();

  /// Called at beginning of loop
  virtual void startingNewLoop( unsigned int iLoop );

  /// Called at end of loop
  virtual Status endOfLoop( const edm::EventSetup&, unsigned int iLoop );

  /// Called at each event 
  virtual Status duringLoop( const edm::Event&, const edm::EventSetup& );

 private:

  // private member functions

  /// Apply random shifts and rotations to selected alignables, according to configuration
  void simpleMisalignment_(const Alignables &alivec, const std::string &selection,
                          float shift, float rot, bool local);

  /// Create tracker and muon geometries
  void createGeometries_( const edm::EventSetup& );

  /// Add survey info to an alignable
  void addSurveyInfo_(
		      Alignable*
		      );

  // private data members

  unsigned int        theSurveyIndex;
  const Alignments*   theSurveyValues;
  const SurveyErrors* theSurveyErrors;

  AlignmentAlgorithmBase* theAlignmentAlgo;
  std::vector<AlignmentMonitorBase*> theMonitors;
  AlignmentParameterStore* theAlignmentParameterStore;

  AlignableTracker* theAlignableTracker;
  AlignableMuon* theAlignableMuon;
  edm::ESHandle<GeometricDet> theGeometricDet; // Needed for AlignableTracker 

  boost::shared_ptr<TrackerGeometry> theTracker;
  boost::shared_ptr<DTGeometry> theMuonDT;
  boost::shared_ptr<CSCGeometry> theMuonCSC;

  int nevent_;

  edm::ParameterSet theParameterSet;

  // steering parameters

  unsigned int theMaxLoops;     // Number of loops to loop

  int stNFixAlignables_;
  double stRandomShift_,stRandomRotation_;
  bool applyDbAlignment_,doMisalignmentScenario_,saveToDB_;
  bool doTracker_,doMuon_;
  bool useSurvey_; // true to read survey info from DB
};

#endif
