#include "PTrackerParametersDBBuilder.h"

#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerParametersFromDD.h"

void
PTrackerParametersDBBuilder::beginRun( const edm::Run&, edm::EventSetup const& es ) 
{
  PTrackerParameters* ptp = new PTrackerParameters;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( !mydbservice.isAvailable())
  {
    edm::LogError( "PTrackerParametersDBBuilder" ) << "PoolDBOutputService unavailable";
    return;
  }
  edm::ESTransientHandle<DDCompactView> cpv;
  es.get<IdealGeometryRecord>().get( cpv );

  TrackerParametersFromDD builder;
  builder.build( &(*cpv), *ptp );
  
  if( mydbservice->isNewTagRequest( "PTrackerParametersRcd" ))
  {
    mydbservice->createNewIOV<PTrackerParameters>( ptp, mydbservice->beginOfTime(), mydbservice->endOfTime(), "PTrackerParametersRcd" );
  } else
  {
    edm::LogError( "PTrackerParametersDBBuilder" ) << "PTrackerParameters and PTrackerParametersRcd Tag already present";
  }
}
