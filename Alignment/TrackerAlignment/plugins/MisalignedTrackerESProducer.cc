// Framework
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Conditions database
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// Alignment
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/TrackerAlignment/interface/TrackerScenarioBuilder.h"
#include "Alignment/CommonAlignment/interface/Alignable.h" 

// C++
#include <boost/shared_ptr.hpp>
#include <memory>
#include <algorithm>

///
/// An ESProducer that fills the TrackerDigiGeometryRcd with a misaligned tracker
/// 
/// This should replace the standard TrackerDigiGeometryESModule when producing
/// Misalignment scenarios.
///

class MisalignedTrackerESProducer: public edm::ESProducer
{
public:

  /// Constructor 
  MisalignedTrackerESProducer(const edm::ParameterSet & p);
  
  /// Destructor
  virtual ~MisalignedTrackerESProducer(); 
  
  /// Produce the misaligned tracker geometry and store it
  boost::shared_ptr<TrackerGeometry> produce(const TrackerDigiGeometryRecord& iRecord);

private:
  const bool theSaveToDB; /// whether or not writing to DB
  const bool theSaveFakeScenario; /// if theSaveToDB is true, save a fake scenario (empty alignments), irrespective of the misalignment scenario below
  const edm::ParameterSet theScenario; /// misalignment scenario
  const edm::ParameterSet thePSet;
  const std::string theAlignRecordName, theErrorRecordName;
  
  boost::shared_ptr<TrackerGeometry> theTracker;
};

//__________________________________________________________________________________________________
//__________________________________________________________________________________________________
//__________________________________________________________________________________________________



//__________________________________________________________________________________________________
MisalignedTrackerESProducer::MisalignedTrackerESProducer(const edm::ParameterSet& p) :
  theSaveToDB(p.getUntrackedParameter<bool>("saveToDbase")),
  theSaveFakeScenario(p.getUntrackedParameter<bool>("saveFakeScenario")),
  theScenario(p.getParameter<edm::ParameterSet>("scenario")),
  thePSet(p),
  theAlignRecordName("TrackerAlignmentRcd"),
  theErrorRecordName("TrackerAlignmentErrorRcd")
{
  setWhatProduced(this);

}


//__________________________________________________________________________________________________
MisalignedTrackerESProducer::~MisalignedTrackerESProducer() {}


//__________________________________________________________________________________________________
boost::shared_ptr<TrackerGeometry> 
MisalignedTrackerESProducer::produce( const TrackerDigiGeometryRecord& iRecord )
{ 
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iRecord.getRecord<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  edm::LogInfo("MisalignedTracker") << "Producer called";

  // Create the tracker geometry from ideal geometry
  edm::ESHandle<GeometricDet> gD;
  iRecord.getRecord<IdealGeometryRecord>().get( gD );
  TrackerGeomBuilderFromGeometricDet trackerBuilder;
  theTracker  = boost::shared_ptr<TrackerGeometry>( trackerBuilder.build(&(*gD), thePSet));
 
  // Create the alignable hierarchy
  std::auto_ptr<AlignableTracker> theAlignableTracker(new AlignableTracker( &(*theTracker), tTopo ) );

  // Create misalignment scenario, apply to geometry
  TrackerScenarioBuilder scenarioBuilder( &(*theAlignableTracker) );
  scenarioBuilder.applyScenario( theScenario );
  Alignments* alignments =  theAlignableTracker->alignments();
  AlignmentErrors* alignmentErrors = theAlignableTracker->alignmentErrors();
  
  // Store result to EventSetup
  GeometryAligner aligner;
  aligner.applyAlignments<TrackerGeometry>( &(*theTracker), alignments, alignmentErrors, 
                                            AlignTransform()); // dummy global position

  // Write alignments to DB: have to sort beforhand!
  if (theSaveToDB) {

      // Call service
      edm::Service<cond::service::PoolDBOutputService> poolDbService;
      if( !poolDbService.isAvailable() ) // Die if not available
        throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";
      if (theSaveFakeScenario) { // make empty!
        alignments->clear();
        alignmentErrors->clear();
      }      
      poolDbService->writeOne<Alignments>(alignments, poolDbService->currentTime(),
                                          theAlignRecordName);
      poolDbService->writeOne<AlignmentErrors>(alignmentErrors, poolDbService->currentTime(),
                                               theErrorRecordName);
  } else {
    // poolDbService::writeOne takes over ownership
    // we have to delete in the case that containers are not written
    delete alignments;
    delete alignmentErrors;
  }
  

  edm::LogInfo("MisalignedTracker") << "Producer done";
  return theTracker;
  
}


DEFINE_FWK_EVENTSETUP_MODULE(MisalignedTrackerESProducer);
