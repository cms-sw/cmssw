#include "RecoLocalTracker/SiStripRecHitConverter/plugins/StripCPEfromTrackAngle2ESProducer.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEfromTrackAngle2.h"
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

StripCPEfromTrackAngle2ESProducer::StripCPEfromTrackAngle2ESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
  cout<<" StripCPEfromTrackAngle2ESProducer constructor "<<endl;
}

StripCPEfromTrackAngle2ESProducer::~StripCPEfromTrackAngle2ESProducer() {}

boost::shared_ptr<StripClusterParameterEstimator> 
StripCPEfromTrackAngle2ESProducer::produce(const TkStripCPERecord & iRecord){ 
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

  _cpe  = boost::shared_ptr<StripClusterParameterEstimator>(new StripCPEfromTrackAngle2(pset_,magfield.product(), pDD.product(),SiStripLorentzAngle_.product()));
  return _cpe;


}


