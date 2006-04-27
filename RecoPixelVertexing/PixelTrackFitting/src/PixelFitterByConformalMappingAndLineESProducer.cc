#include "PixelFitterByConformalMappingAndLineESProducer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

using namespace edm;

PixelFitterByConformalMappingAndLineESProducer::PixelFitterByConformalMappingAndLineESProducer(const edm::ParameterSet & p) 
  : theConfig(p)
{
//  std::string myname = p.getParameter<std::string>("ComponentName");
  std::string myname = "PixelFitterByConformalMappingAndLine";
  setWhatProduced(this,myname);

}

PixelFitterByConformalMappingAndLineESProducer::~PixelFitterByConformalMappingAndLineESProducer()
{
}

  boost::shared_ptr<PixelFitter> PixelFitterByConformalMappingAndLineESProducer::produce(const TrackingComponentsRecord & r)
{
  theFitter = boost::shared_ptr<PixelFitter>(new PixelFitterByConformalMappingAndLine());
  return theFitter;
}
