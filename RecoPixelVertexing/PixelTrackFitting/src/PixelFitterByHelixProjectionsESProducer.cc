#include "PixelFitterByHelixProjectionsESProducer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

using namespace edm;

PixelFitterByHelixProjectionsESProducer::PixelFitterByHelixProjectionsESProducer(const edm::ParameterSet & p) 
  : theConfig(p)
{
  std::string myname = "PixelFitterByHelixProjections";
  setWhatProduced(this,myname);
}

PixelFitterByHelixProjectionsESProducer::~PixelFitterByHelixProjectionsESProducer()
{
}

boost::shared_ptr<PixelFitter> PixelFitterByHelixProjectionsESProducer::produce(const TrackingComponentsRecord & r)
{
  theFitter = boost::shared_ptr<PixelFitter>(new PixelFitterByHelixProjections());
  return theFitter;
}
