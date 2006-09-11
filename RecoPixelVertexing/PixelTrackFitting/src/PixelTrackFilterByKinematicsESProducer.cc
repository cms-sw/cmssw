#include "PixelTrackFilterByKinematicsESProducer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

using namespace edm;

PixelTrackFilterByKinematicsESProducer::PixelTrackFilterByKinematicsESProducer(const edm::ParameterSet & p)
  : theConfig(p)
{
  std::string myname = "PixelTrackFilterByKinematics";
  setWhatProduced(this,myname);
}

PixelTrackFilterByKinematicsESProducer::~PixelTrackFilterByKinematicsESProducer()
{
}

boost::shared_ptr<PixelTrackFilter> PixelTrackFilterByKinematicsESProducer::produce(const TrackingComponentsRecord & r)
{
  theFilter = boost::shared_ptr<PixelTrackFilter>(
      new PixelTrackFilterByKinematics( 
          theConfig.getParameter<double>("ptMin"),
          theConfig.getParameter<double>("tipMax"),
          theConfig.getParameter<double>("chi2") ) );
  return theFilter;
}

