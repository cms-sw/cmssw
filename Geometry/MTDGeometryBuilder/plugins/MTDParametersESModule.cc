#include "MTDParametersESModule.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PMTDParametersRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDParametersFromDD.h"
#include "CondFormats/GeometryObjects/interface/PMTDParameters.h"

MTDParametersESModule::MTDParametersESModule( const edm::ParameterSet& pset) 
{
  edm::LogInfo("TRACKER") << "MTDParametersESModule::MTDParametersESModule";

  setWhatProduced(this);
}

MTDParametersESModule::~MTDParametersESModule()
{ 
}

void
MTDParametersESModule::fillDescriptions( edm::ConfigurationDescriptions & descriptions ) 
{
  edm::ParameterSetDescription desc;
  descriptions.add( "mtdParameters", desc );
}

MTDParametersESModule::ReturnType
MTDParametersESModule::produce( const PMTDParametersRcd& iRecord )
{
  edm::LogInfo("MTDParametersESModule") <<  "MTDParametersESModule::produce(const PMTDParametersRcd& iRecord)" << std::endl;
  edm::ESTransientHandle<DDCompactView> cpv;
  iRecord.getRecord<IdealGeometryRecord>().get( cpv );
    
  auto ptp = std::make_unique<PMTDParameters>();
  builder.build( &(*cpv), *ptp );
  
  return ptp;
}

DEFINE_FWK_EVENTSETUP_MODULE( MTDParametersESModule);
