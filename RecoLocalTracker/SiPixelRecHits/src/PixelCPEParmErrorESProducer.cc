#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEParmErrorESProducer.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEParmError.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"



#include <string>
#include <memory>

using namespace edm;

PixelCPEParmErrorESProducer::PixelCPEParmErrorESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

PixelCPEParmErrorESProducer::~PixelCPEParmErrorESProducer() {}

boost::shared_ptr<PixelClusterParameterEstimator> 
PixelCPEParmErrorESProducer::produce(const TrackerCPERecord & iRecord){ 
//   if (_propagator){
//     delete _propagator;
//     _propagator = 0;
//   }
  ESHandle<MagneticField> magfield;
  iRecord.getRecord<IdealMagneticFieldRecord>().get(magfield );

  edm::ESHandle<TrackerGeometry> pDD;
  iRecord.getRecord<TrackerDigiGeometryRecord>().get( pDD );

  //  cpe_  = boost::shared_ptr<PixelClusterParameterEstimator>(new PixelCPEParmError(pset_,magfield.product(), pDD.product()));
  cpe_  = boost::shared_ptr<PixelClusterParameterEstimator>(new PixelCPEParmError(pset_,magfield.product()) );
  return cpe_;
}


