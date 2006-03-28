#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEfromTrackAngleESProducer.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEfromTrackAngle.h"
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

StripCPEfromTrackAngleESProducer::StripCPEfromTrackAngleESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

StripCPEfromTrackAngleESProducer::~StripCPEfromTrackAngleESProducer() {}

boost::shared_ptr<StripClusterParameterEstimator> 
StripCPEfromTrackAngleESProducer::produce(const TrackerCPERecord & iRecord){ 
//   if (_propagator){
//     delete _propagator;
//     _propagator = 0;
//   }
  ESHandle<MagneticField> magfield;
  iRecord.getRecord<IdealMagneticFieldRecord>().get(magfield );

  edm::ESHandle<TrackerGeometry> pDD;
  iRecord.getRecord<TrackerDigiGeometryRecord>().get( pDD );

  _cpe  = boost::shared_ptr<StripClusterParameterEstimator>(new StripCPEfromTrackAngle(pset_,magfield.product(), pDD.product()));
  return _cpe;
}


