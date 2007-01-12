#ifndef Alignment_CommonAlignmentAlgorithm_TrackerAlignmentProducer_h
#define Alignment_CommonAlignmentAlgorithm_TrackerAlignmentProducer_h

/// \class AlignmentProducer
///
/// Package     : Alignment/CommonAlignmentProducer
/// Description : calls alignment algorithms
///
///  \author    : Frederic Ronga
///  Revision   : $Revision: 1.8 $
///  last update: $Date: 2006/11/07 10:33:46 $
///  by         : $Author: flucke $

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
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterBuilder.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"


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

  // Define return type
  typedef boost::shared_ptr<TrackerGeometry> ReturnType;

  /// Produce the geometry
  virtual ReturnType produce( const TrackerDigiGeometryRecord& iRecord );

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
  void simpleMisalignment(const Alignables &alivec, const std::string &selection,
                          float shift, float rot, bool local);

  /// Check if a trajectory and a track match
  const bool trajTrackMatch( const ConstTrajTrackPair& pair ) const;

  // private data members

  AlignmentAlgorithmBase* theAlignmentAlgo;
  AlignmentParameterBuilder* theAlignmentParameterBuilder;
  AlignmentParameterStore* theAlignmentParameterStore;
  AlignableTracker* theAlignableTracker;

  ReturnType theTracker;
  int nevent;

  edm::ParameterSet theParameterSet;

  // steering parameters

  unsigned int theMaxLoops;     // Number of loops to loop

  int stNFixAlignables;
  double stRandomShift,stRandomRotation;
  bool applyDbAlignment_,doMisalignmentScenario,saveToDB;

};

#endif
