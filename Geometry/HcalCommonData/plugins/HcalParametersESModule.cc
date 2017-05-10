#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "CondFormats/GeometryObjects/interface/PHcalParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PHcalParametersRcd.h"
#include "Geometry/HcalCommonData/interface/HcalParametersFromDD.h"

#include <boost/shared_ptr.hpp>
 
namespace edm {
  class ConfigurationDescriptions;
}
class PHcalParameters;
class PHcalParametersRcd;

class  HcalParametersESModule : public edm::ESProducer
{
 public:
  HcalParametersESModule( const edm::ParameterSet & );
  ~HcalParametersESModule( void );
  
  typedef boost::shared_ptr<PHcalParameters> ReturnType;

  static void fillDescriptions( edm::ConfigurationDescriptions & );
  
  ReturnType produce( const PHcalParametersRcd & );
};

HcalParametersESModule::HcalParametersESModule( const edm::ParameterSet& )
{
  edm::LogInfo("HCAL") << "HcalParametersESModule::HcalParametersESModule";

  setWhatProduced(this);
}

HcalParametersESModule::~HcalParametersESModule()
{}

void
HcalParametersESModule::fillDescriptions( edm::ConfigurationDescriptions & descriptions ) 
{
  edm::ParameterSetDescription desc;
  descriptions.add( "hcalParameters", desc );
}

HcalParametersESModule::ReturnType
HcalParametersESModule::produce( const PHcalParametersRcd& iRecord )
{
  //edm::LogInfo("HcalParametersESModule")
  std::cout <<  "HcalParametersESModule::produce(const PHcalParametersRcd& iRecord)" << std::endl;
  edm::ESTransientHandle<DDCompactView> cpv;
  iRecord.getRecord<IdealGeometryRecord>().get( cpv );
  
  PHcalParameters* ptp = new PHcalParameters();
  HcalParametersFromDD builder;
  builder.build( &(*cpv), *ptp );
  
  return ReturnType( ptp ) ;
}

DEFINE_FWK_EVENTSETUP_MODULE( HcalParametersESModule);
