#include "RecoLocalTracker/SiStripRecHitConverter/plugins/StripCPEfromTrackAngleESProducer.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEfromTrackAngle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"

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
StripCPEfromTrackAngleESProducer::produce(const TkStripCPERecord & iRecord){ 
//   if (_propagator){
//     delete _propagator;
//     _propagator = 0;
//   }
  ESHandle<MagneticField> magfield;
  iRecord.getRecord<IdealMagneticFieldRecord>().get(magfield );

  edm::ESHandle<TrackerGeometry> pDD;
  iRecord.getRecord<TrackerDigiGeometryRecord>().get( pDD );

  edm::ESHandle<SiStripLorentzAngle> SiStripLorentzAngle_;
  iRecord.getRecord<SiStripLorentzAngleRcd>().get(SiStripLorentzAngle_);
  _cpe  = boost::shared_ptr<StripClusterParameterEstimator>(new StripCPEfromTrackAngle(pset_,magfield.product(), pDD.product(),SiStripLorentzAngle_.product() ));

  return _cpe;
}


