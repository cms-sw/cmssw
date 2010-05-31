#include "RecoLocalTracker/SiStripRecHitConverter/plugins/StripCPEESProducer.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEfromTrackAngle.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEfromTrackAngle2.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEgeometric.h"
#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "FWCore/Framework/interface/ESHandle.h"

StripCPEESProducer::StripCPEESProducer(const edm::ParameterSet & p) 
{
  name_ = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,name_);
}

boost::shared_ptr<StripClusterParameterEstimator> StripCPEESProducer::
produce(const TkStripCPERecord & iRecord) 
{ 
  edm::ESHandle<TrackerGeometry> pDD;  iRecord.getRecord<TrackerDigiGeometryRecord>().get( pDD );
  edm::ESHandle<MagneticField> magfield;  iRecord.getRecord<IdealMagneticFieldRecord>().get(magfield );
  edm::ESHandle<SiStripLorentzAngle> lorentzAngle;  iRecord.getRecord<SiStripLorentzAngleRcd>().get(lorentzAngle);
  edm::ESHandle<SiStripConfObject> confObj;  iRecord.getRecord<SiStripConfObjectRcd>().get(confObj);
  edm::ESHandle<SiStripLatency> latency;  iRecord.getRecord<SiStripLatencyRcd>().get(latency);
  edm::ESHandle<SiStripNoises> noise;  iRecord.getRecord<SiStripNoisesRcd>().get(noise);
  edm::ESHandle<SiStripApvGain> gain;  iRecord.getRecord<SiStripApvGainRcd>().get(gain);
  edm::ESHandle<SiStripBadStrip> bad;  iRecord.getRecord<SiStripBadChannelRcd>().get(bad);

  if(name_=="SimpleStripCPE") 
    cpe_ = boost::shared_ptr<StripClusterParameterEstimator>(new StripCPE(               pset_, magfield.product(), pDD.product(), lorentzAngle.product() ));  
  else if(name_=="StripCPEfromTrackAngle")
    cpe_ = boost::shared_ptr<StripClusterParameterEstimator>(new StripCPEfromTrackAngle( pset_, magfield.product(), pDD.product(), lorentzAngle.product() ));
  else if(name_=="StripCPEfromTrackAngle2")
    cpe_ = boost::shared_ptr<StripClusterParameterEstimator>(new StripCPEfromTrackAngle2(pset_, magfield.product(), pDD.product(), lorentzAngle.product() ));
  else if(name_=="StripCPEgeometric")
    cpe_ = boost::shared_ptr<StripClusterParameterEstimator>(new StripCPEgeometric(pset_, magfield.product(), pDD.product(), lorentzAngle.product() ));
  else throw cms::Exception("Unknown StripCPE type") << name_;

  return cpe_;
}
