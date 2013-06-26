// -*- C++ -*-
//
// Package:    SiStripLorentzAngleDepESProducer
// Class:      SiStripLorentzAngleDepESProducer
// 
/**\class SiStripLorentzAngleDepESProducer SiStripLorentzAngleDepESProducer.h CalibTracker/SiStripESProducers/plugins/real/SiStripLorentzAngleDepESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Michael Segala and Rebeca Gonzalez Suarez 
//         Created:  15/02/2011
// $Id: SiStripLorentzAngleDepESProducer.cc,v 1.2 2011/03/16 13:22:55 rebeca Exp $
//
//



#include "CalibTracker/SiStripESProducers/plugins/real/SiStripLorentzAngleDepESProducer.h"


SiStripLorentzAngleDepESProducer::SiStripLorentzAngleDepESProducer(const edm::ParameterSet& iConfig):
  pset_(iConfig),
  getLatency(iConfig.getParameter<edm::ParameterSet>("LatencyRecord")),
  getPeak(iConfig.getParameter<edm::ParameterSet>("LorentzAnglePeakMode")),
  getDeconv(iConfig.getParameter<edm::ParameterSet>("LorentzAngleDeconvMode"))
{  
  setWhatProduced(this);
  
  edm::LogInfo("SiStripLorentzAngleDepESProducer") << "ctor" << std::endl;

}


boost::shared_ptr<SiStripLorentzAngle> SiStripLorentzAngleDepESProducer::produce(const SiStripLorentzAngleDepRcd& iRecord)
{

  edm::LogInfo("SiStripLorentzAngleDepESProducer") << "Producer called" << std::endl;
  
  std::string latencyRecordName = getLatency.getParameter<std::string>("record");
  std::string latencyLabel = getLatency.getUntrackedParameter<std::string>("label");
  bool peakMode = false;
  
  if( latencyRecordName == "SiStripLatencyRcd" ) {      
    edm::ESHandle<SiStripLatency> latency;  
    iRecord.getRecord<SiStripLatencyRcd>().get( latencyLabel, latency);
    if(latency -> singleReadOutMode() == 1) peakMode = true;
  } else edm::LogError("SiStripLorentzAngleDepESProducer") << "[SiStripLorentzAngleDepESProducer::produce] No Latency Record found " << std::endl;
 
  std::string lorentzAngleRecordName;
  std::string lorentzAngleLabel;
  	 
  if (peakMode){
    lorentzAngleRecordName = getPeak.getParameter<std::string>("record");
    lorentzAngleLabel = getPeak.getUntrackedParameter<std::string>("label"); 
  } else {
    lorentzAngleRecordName = getDeconv.getParameter<std::string>("record");
    lorentzAngleLabel = getDeconv.getUntrackedParameter<std::string>("label"); 
  } 
  
  if ( lorentzAngleRecordName == "SiStripLorentzAngleRcd"){
    edm::ESHandle<SiStripLorentzAngle> siStripLorentzAngle;
    iRecord.getRecord<SiStripLorentzAngleRcd>().get(lorentzAngleLabel, siStripLorentzAngle);
    siStripLA_.reset(new SiStripLorentzAngle(*(siStripLorentzAngle.product())));
  } else edm::LogError("SiStripLorentzAngleDepESProducer") << "[SiStripLorentzAngleDepESProducer::produce] No Lorentz Angle Record found " << std::endl;
	 

   return siStripLA_;

  
}

