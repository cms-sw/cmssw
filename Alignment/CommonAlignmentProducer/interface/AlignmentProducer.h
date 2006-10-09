#ifndef Alignment_CommonAlignmentAlgorithm_TrackerAlignmentProducer_h
#define Alignment_CommonAlignmentAlgorithm_TrackerAlignmentProducer_h

//
// Package:    Alignment/CommonAlignmentAlgorithm
// Class:      AlignmentProducer
// 
//
// Description: calls alignment algorithms
//
//
// Original Author:  Frederic Ronga
//

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

#include "TFile.h"
#include "TTree.h"


class AlignmentProducer : public edm::ESProducerLooper
{

 public:

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

  typedef std::vector<Alignable*> Alignables;

  // private member functions

  void simpleMisalignment(Alignables alivec, std::vector<bool>sel, 
    float shift, float rot, bool local);

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

  std::string stParameterSelector;
  std::string stAlignableSelector;
  int stNFixAlignables;
  double stRandomShift,stRandomRotation;
  bool doMisalignmentScenario,saveToDB;

};

#endif
