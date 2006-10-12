// Framework
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Conditions database
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"

// Alignment
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/AlignmentSorter.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/TrackerAlignment/interface/MisalignmentScenarioBuilder.h"

#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"

#include "Alignment/TrackerAlignment/interface/MisalignedTrackerESProducer.h"

///
/// An ESProducer that fills the TrackerDigiGeometryRcd with a misaligned tracker
/// 
/// This should replace the standard TrackerDigiGeometryESModule when producing
/// Misalignment scenarios.

#include <memory>
#include <algorithm>

//__________________________________________________________________________________________________
MisalignedTrackerESProducer::MisalignedTrackerESProducer(const edm::ParameterSet& p) :
  theParameterSet( p )
{
  
  setWhatProduced(this);

}


//__________________________________________________________________________________________________
MisalignedTrackerESProducer::~MisalignedTrackerESProducer() {}


//__________________________________________________________________________________________________
boost::shared_ptr<TrackerGeometry> 
MisalignedTrackerESProducer::produce( const TrackerDigiGeometryRecord& iRecord )
{ 

  edm::LogInfo("MisalignedTracker") << "Producer called";

  // Create the tracker geometry from ideal geometry
  edm::ESHandle<DDCompactView> cpv;
  edm::ESHandle<GeometricDet> gD;
  iRecord.getRecord<IdealGeometryRecord>().get( cpv );
  iRecord.getRecord<IdealGeometryRecord>().get( gD );
  TrackerGeomBuilderFromGeometricDet trackerBuilder;
  theTracker  = boost::shared_ptr<TrackerGeometry>( trackerBuilder.build(&(*cpv),&(*gD)) );
  
  // Create the alignable hierarchy
  AlignableTracker* theAlignableTracker = new AlignableTracker( &(*gD), &(*theTracker) );

  // Create misalignment scenario, apply to geometry
  MisalignmentScenarioBuilder scenarioBuilder( theAlignableTracker );
  scenarioBuilder.applyScenario( theParameterSet );
  Alignments* alignments =  theAlignableTracker->alignments();
  AlignmentErrors* alignmentErrors = theAlignableTracker->alignmentErrors();
  
  // Write alignments to DB: have to sort beforhand!
  if ( theParameterSet.getUntrackedParameter<bool>("saveToDbase", false) )
	{

	  // Call service
	  edm::Service<cond::service::PoolDBOutputService> poolDbService;
	  if( !poolDbService.isAvailable() ) // Die if not available
		throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";
	  
	  // Define callback tokens for the two records
	  size_t alignmentsToken = poolDbService->callbackToken("Alignments");
	  size_t alignmentErrorsToken = poolDbService->callbackToken("AlignmentErrors");
	  
	  // Store
	  poolDbService->newValidityForNewPayload<Alignments>( &(*alignments), 
														   poolDbService->endOfTime(),
														   alignmentsToken );
	  poolDbService->newValidityForNewPayload<AlignmentErrors>( &(*alignmentErrors), 
																poolDbService->endOfTime(),
																alignmentErrorsToken );
	}
  
  // Store result to EventSetup
  GeometryAligner aligner;
  aligner.applyAlignments<TrackerGeometry>( &(*theTracker), &(*alignments), &(*alignmentErrors) );

  edm::LogInfo("MisalignedTracker") << "Producer done";
  return theTracker;
  
}


DEFINE_FWK_EVENTSETUP_MODULE(MisalignedTrackerESProducer)
