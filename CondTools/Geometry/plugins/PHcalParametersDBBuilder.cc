#include "PHcalParametersDBBuilder.h"

#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/GeometryObjects/interface/PHcalParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HcalTowerAlgo/interface/HcalParametersFromDD.h"

void
PHcalParametersDBBuilder::beginRun( const edm::Run&, edm::EventSetup const& es ) 
{
  PHcalParameters* php = new PHcalParameters;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( !mydbservice.isAvailable())
  {
    edm::LogError( "PHcalParametersDBBuilder" ) << "PoolDBOutputService unavailable";
    return;
  }
  edm::ESTransientHandle<DDCompactView> cpv;
  es.get<IdealGeometryRecord>().get( cpv );
  
  HcalParametersFromDD builder;
  builder.build( &(*cpv), *php );
  
  if( mydbservice->isNewTagRequest( "PHcalParametersRcd" ))
  {
    mydbservice->createNewIOV<PHcalParameters>( php, mydbservice->beginOfTime(), mydbservice->endOfTime(), "PHcalParametersRcd" );
  } else
  {
    edm::LogError( "PHcalParametersDBBuilder" ) << "PHcalParameters and PHcalParametersRcd Tag already present";
  }
}
