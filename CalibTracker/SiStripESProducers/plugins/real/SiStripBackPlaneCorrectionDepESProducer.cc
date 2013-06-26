// -*- C++ -*-
//
// Package:    SiStripBackPlaneCorrectionDepESProducer
// Class:      SiStripBackPlaneCorrectionDepESProducer
// 
/**\class SiStripBackPlaneCorrectionDepESProducer SiStripBackPlaneCorrectionDepESProducer.h CalibTracker/SiStripESProducers/plugins/real/SiStripBackPlaneCorrectionDepESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Loic Quertenmont inspired from the SiStripLorentzAngleDepESProducer 
//         Created:  07/03/2013
// $Id: SiStripBackPlaneCorrectionDepESProducer.cc,v 1.1 2013/03/15 11:24:03 querten Exp $
//
//

#include "CalibTracker/SiStripESProducers/plugins/real/SiStripBackPlaneCorrectionDepESProducer.h"


SiStripBackPlaneCorrectionDepESProducer::SiStripBackPlaneCorrectionDepESProducer(const edm::ParameterSet& iConfig):
  pset_(iConfig),
  getLatency(iConfig.getParameter<edm::ParameterSet>("LatencyRecord")),
  getPeak(iConfig.getParameter<edm::ParameterSet>("BackPlaneCorrectionPeakMode")),
  getDeconv(iConfig.getParameter<edm::ParameterSet>("BackPlaneCorrectionDeconvMode"))
{  
  setWhatProduced(this);
  
  edm::LogInfo("SiStripBackPlaneCorrectionDepESProducer") << "ctor" << std::endl;

}


boost::shared_ptr<SiStripBackPlaneCorrection> SiStripBackPlaneCorrectionDepESProducer::produce(const SiStripBackPlaneCorrectionDepRcd& iRecord)
{

  edm::LogInfo("SiStripBackPlaneCorrectionDepESProducer") << "Producer called" << std::endl;
  
  std::string latencyRecordName = getLatency.getParameter<std::string>("record");
  std::string latencyLabel = getLatency.getUntrackedParameter<std::string>("label");
  bool peakMode = false;
  
  if( latencyRecordName == "SiStripLatencyRcd" ) {      
    edm::ESHandle<SiStripLatency> latency;  
    iRecord.getRecord<SiStripLatencyRcd>().get( latencyLabel, latency);
    if(latency -> singleReadOutMode() == 1) peakMode = true;
  } else edm::LogError("SiStripBackPlaneCorrectionDepESProducer") << "[SiStripBackPlaneCorrectionDepESProducer::produce] No Latency Record found " << std::endl;
 
  std::string backPlaneCorrectionRecordName;
  std::string backPlaneCorrectionLabel;
  	 
  if (peakMode){
    backPlaneCorrectionRecordName = getPeak.getParameter<std::string>("record");
    backPlaneCorrectionLabel = getPeak.getUntrackedParameter<std::string>("label"); 
  } else {
    backPlaneCorrectionRecordName = getDeconv.getParameter<std::string>("record");
    backPlaneCorrectionLabel = getDeconv.getUntrackedParameter<std::string>("label"); 
  } 
  
  if ( backPlaneCorrectionRecordName == "SiStripBackPlaneCorrectionRcd"){
    edm::ESHandle<SiStripBackPlaneCorrection> siStripBackPlaneCorrection;
    iRecord.getRecord<SiStripBackPlaneCorrectionRcd>().get(backPlaneCorrectionLabel, siStripBackPlaneCorrection);
    siStripBPC_.reset(new SiStripBackPlaneCorrection(*(siStripBackPlaneCorrection.product())));
  } else edm::LogError("SiStripBackPlaneCorrectionDepESProducer") << "[SiStripBackPlaneCorrectionDepESProducer::produce] No Lorentz Angle Record found " << std::endl;
	 

   return siStripBPC_;

  
}

