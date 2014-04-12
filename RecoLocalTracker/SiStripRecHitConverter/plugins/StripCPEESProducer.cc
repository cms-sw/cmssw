#include "RecoLocalTracker/SiStripRecHitConverter/plugins/StripCPEESProducer.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEfromTrackAngle.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEfromTemplate.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEgeometric.h"
#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"
#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "FWCore/Framework/interface/ESHandle.h"

StripCPEESProducer::StripCPEESProducer(const edm::ParameterSet & p) 
{
  std::string name = p.getParameter<std::string>("ComponentName");
  std::string type=name;
  if (!p.exists("ComponentType"))
    edm::LogWarning("StripCPEESProducer")<<" the CPE producer should contain a ComponentType, probably identical to ComponentName in the first step of migration. Falling back to:"<<type;
  else
    type=p.getParameter<std::string>("ComponentType");

  enumMap[std::string("SimpleStripCPE")]=SIMPLE;
  enumMap[std::string("StripCPEfromTrackAngle")]=TRACKANGLE;
  enumMap[std::string("StripCPEgeometric")]=GEOMETRIC;
  enumMap[std::string("StripCPEfromTemplate")]=TEMPLATE;
  if(enumMap.find(type)==enumMap.end()) 
    throw cms::Exception("Unknown StripCPE type") << type;

  cpeNum = enumMap[type];
  pset = p;
  setWhatProduced(this,name);
}

boost::shared_ptr<StripClusterParameterEstimator> StripCPEESProducer::
produce(const TkStripCPERecord & iRecord) 
{ 
  edm::ESHandle<TrackerGeometry> pDD;  iRecord.getRecord<TrackerDigiGeometryRecord>().get( pDD );
  edm::ESHandle<MagneticField> magfield;  iRecord.getRecord<IdealMagneticFieldRecord>().get(magfield );
  edm::ESHandle<SiStripLorentzAngle> lorentzAngle;   iRecord.getRecord<SiStripLorentzAngleDepRcd>().get(lorentzAngle);
  edm::ESHandle<SiStripBackPlaneCorrection> backPlaneCorrection; iRecord.getRecord<SiStripBackPlaneCorrectionDepRcd>().get(backPlaneCorrection);
  edm::ESHandle<SiStripConfObject> confObj;  iRecord.getRecord<SiStripConfObjectRcd>().get(confObj);
  edm::ESHandle<SiStripLatency> latency;  iRecord.getRecord<SiStripLatencyRcd>().get(latency);
  edm::ESHandle<SiStripNoises> noise;  iRecord.getRecord<SiStripNoisesRcd>().get(noise);
  edm::ESHandle<SiStripApvGain> gain;  iRecord.getRecord<SiStripApvGainRcd>().get(gain);
  edm::ESHandle<SiStripBadStrip> bad;  iRecord.getRecord<SiStripBadChannelRcd>().get(bad);
 
  
  switch(cpeNum) {

  case SIMPLE:     
    cpe = boost::shared_ptr<StripClusterParameterEstimator>(new StripCPE( pset, *magfield, *pDD, *lorentzAngle, *backPlaneCorrection, *confObj, *latency ));  
    break;
    
  case TRACKANGLE: 
    cpe = boost::shared_ptr<StripClusterParameterEstimator>(new StripCPEfromTrackAngle( pset, *magfield, *pDD, *lorentzAngle, *backPlaneCorrection, *confObj, *latency )); 
    break;
    
  case GEOMETRIC:  
    cpe = boost::shared_ptr<StripClusterParameterEstimator>(new StripCPEgeometric(pset, *magfield, *pDD, *lorentzAngle, *backPlaneCorrection, *confObj, *latency )); 
    break;  

  case TEMPLATE: 
    cpe = boost::shared_ptr<StripClusterParameterEstimator>(new StripCPEfromTemplate( pset, *magfield, *pDD, *lorentzAngle, *backPlaneCorrection, *confObj, *latency )); 
    break;


  }

  return cpe;
}
