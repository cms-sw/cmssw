#include "RecoLocalTracker/SiStripRecHitConverter/plugins/StripCPEESProducer.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"
#include "FWCore/Framework/interface/ESHandle.h"

StripCPEESProducer::StripCPEESProducer(const edm::ParameterSet & p) 
{
  std::string name = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,name);
}

boost::shared_ptr<StripClusterParameterEstimator> StripCPEESProducer::
produce(const TkStripCPERecord & iRecord) 
{ 
  edm::ESHandle<MagneticField> magfield;
  iRecord.getRecord<IdealMagneticFieldRecord>().get(magfield );
  
  edm::ESHandle<TrackerGeometry> pDD;
  iRecord.getRecord<TrackerDigiGeometryRecord>().get( pDD );
  
  edm::ESHandle<SiStripLorentzAngle> SiStripLorentzAngle_;
  iRecord.getRecord<SiStripLorentzAngleRcd>().get(SiStripLorentzAngle_);
  cpe_  = boost::shared_ptr<StripClusterParameterEstimator>(new StripCPE(pset_,magfield.product(), pDD.product(), SiStripLorentzAngle_.product()));
  
  return cpe_;
}


