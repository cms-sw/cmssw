#include "RecoLocalTracker/SiStripRecHitConverter/plugins/StripCPEfromTrackAngleESProducer.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEfromTrackAngle.h"
#include "FWCore/Framework/interface/ESHandle.h"

StripCPEfromTrackAngleESProducer::StripCPEfromTrackAngleESProducer(const edm::ParameterSet & p) 
{
  std::string name = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,name);
}

boost::shared_ptr<StripClusterParameterEstimator> StripCPEfromTrackAngleESProducer::
produce(const TkStripCPERecord & iRecord) { 
  edm::ESHandle<MagneticField> magfield;
  iRecord.getRecord<IdealMagneticFieldRecord>().get(magfield );

  edm::ESHandle<TrackerGeometry> pDD;
  iRecord.getRecord<TrackerDigiGeometryRecord>().get( pDD );

  edm::ESHandle<SiStripLorentzAngle> SiStripLorentzAngle_;
  iRecord.getRecord<SiStripLorentzAngleRcd>().get(SiStripLorentzAngle_);
  cpe_  = boost::shared_ptr<StripClusterParameterEstimator>(new StripCPEfromTrackAngle(pset_,magfield.product(), pDD.product(),SiStripLorentzAngle_.product() ));

  return cpe_;
}


