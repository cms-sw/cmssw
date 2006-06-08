// Framework
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

// Alignment
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"
#include "Alignment/TrackerAlignment/interface/MisalignmentScenarioBuilder.h"

// Class definition
#include "Alignment/TrackerAlignment/interface/MisalignedTrackerESProducer.h"

///
/// An ESProducer that fills the TrackerDigiGeometryRcd with a misaligned tracker
/// 
/// This should replace the standard TrackerDigiGeometryESModule when producing
/// Misalignment scenarios.

#include <memory>


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
  
  // Dump alignments BEFORE
  Alignments* alignments;
  if ( theParameterSet.getUntrackedParameter<bool>("dumpBefore", false) )
	{
	  alignments = theAlignableTracker->alignments();
	  for ( std::vector<AlignTransform>::iterator it = alignments->m_align.begin();
			it != alignments->m_align.end(); it++ )
		std::cout << (*it).rawId() << " " << (*it).translation() << std::endl;
	}

  // Create misalignment scenario
  MisalignmentScenarioBuilder scenarioBuilder( theAlignableTracker );
  scenarioBuilder.applyScenario( theParameterSet );


  // Dump alignments AFTER
  if ( theParameterSet.getUntrackedParameter<bool>("dumpAfter", false) )
	{
	  alignments = theAlignableTracker->alignments();
	  for ( std::vector<AlignTransform>::iterator it = alignments->m_align.begin();
			it != alignments->m_align.end(); it++ )
		std::cout << (*it).rawId() << " " << (*it).translation() << std::endl;
	}

  // Write alignments to DB
  if ( theParameterSet.getUntrackedParameter<bool>("saveToDbase", false) )
	{
	  edm::Service<cond::service::PoolDBOutputService> poolDbService;
	  if( poolDbService.isAvailable() )
		{
		  alignments = theAlignableTracker->alignments();
		  poolDbService->newValidityForNewPayload<Alignments>( alignments, 
															   poolDbService->endOfTime() );
		}
	}
  
  edm::LogInfo("MisalignedTracker") << "Producer done";

  // Store result to EventSetup
  return theTracker;
  
}


DEFINE_FWK_EVENTSETUP_MODULE(MisalignedTrackerESProducer)
