#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "CondFormats/GeometryObjects/interface/HcalParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalParametersRcd.h"
#include "Geometry/HcalCommonData/interface/HcalParametersFromDD.h"

#include <memory>
 

class  HcalParametersESModule : public edm::ESProducer {
public:
  HcalParametersESModule( const edm::ParameterSet & );
  ~HcalParametersESModule( void ) override;
  
  using ReturnType = std::unique_ptr<HcalParameters>;

  static void fillDescriptions( edm::ConfigurationDescriptions & );
  
  ReturnType produce( const HcalParametersRcd & );
};

HcalParametersESModule::HcalParametersESModule( const edm::ParameterSet& ) {
  edm::LogInfo("HCAL") << "HcalParametersESModule::HcalParametersESModule";

  setWhatProduced(this);
}

HcalParametersESModule::~HcalParametersESModule() {}

void HcalParametersESModule::fillDescriptions( edm::ConfigurationDescriptions & descriptions ) {
  edm::ParameterSetDescription desc;
  descriptions.add( "hcalParameters", desc );
}

HcalParametersESModule::ReturnType
HcalParametersESModule::produce( const HcalParametersRcd& iRecord ) {
  edm::LogInfo("HcalESModule")
    <<  "HcalParametersESModule::produce(const HcalParametersRcd& iRecord)";
  edm::ESTransientHandle<DDCompactView> cpv;
  iRecord.getRecord<IdealGeometryRecord>().get( cpv );
  
  auto ptp = std::make_unique<HcalParameters>();
  HcalParametersFromDD builder;
  builder.build( &(*cpv), *ptp );
  
  return ptp;
}

DEFINE_FWK_EVENTSETUP_MODULE( HcalParametersESModule);
