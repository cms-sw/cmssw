#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PMTDParametersRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDParametersFromDD.h"
#include "CondFormats/GeometryObjects/interface/PMTDParameters.h"

#include <memory>

class  MTDParametersESModule: public edm::ESProducer
{
 public:
  MTDParametersESModule( const edm::ParameterSet & );

  using ReturnType = std::unique_ptr<PMTDParameters>;

  static void fillDescriptions( edm::ConfigurationDescriptions & );
  
  ReturnType produce( const PMTDParametersRcd & );

 private:
  MTDParametersFromDD builder;
  const edm::ESGetToken<DDCompactView, IdealGeometryRecord> compactViewToken_;
 };

MTDParametersESModule::MTDParametersESModule( const edm::ParameterSet& pset) :
  compactViewToken_{ setWhatProduced(this).consumesFrom<DDCompactView, IdealGeometryRecord>(edm::ESInputTag()) }
{
  edm::LogInfo("TRACKER") << "MTDParametersESModule::MTDParametersESModule";
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
  auto cpv = iRecord.getTransientHandle( compactViewToken_ );
  auto ptp = std::make_unique<PMTDParameters>();
  builder.build( cpv.product(), *ptp );
  
  return ptp;
}

DEFINE_FWK_EVENTSETUP_MODULE( MTDParametersESModule);
