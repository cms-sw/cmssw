// -*- C++ -*-
//
// Package:    SiStripQualityESProducer
// Class:      SiStripQualityESProducer
// 
/**\class SiStripQualityESProducer SiStripQualityESProducer.h CalibTracker/SiStripESProducers/plugins/real/SiStripQualityESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Domenico GIORDANO
//         Created:  Wed Oct  3 12:11:10 CEST 2007
// $Id: SiStripQualityESProducer.cc,v 1.9 2010/02/09 08:39:33 demattia Exp $
//
//



#include "CalibTracker/SiStripESProducers/plugins/real/SiStripQualityESProducer.h"



SiStripQualityESProducer::SiStripQualityESProducer(const edm::ParameterSet& iConfig):
  pset_(iConfig),
  toGet(iConfig.getParameter<Parameters>("ListOfRecordToMerge"))
{
  
  setWhatProduced(this);
  
  edm::LogInfo("SiStripQualityESProducer") << "ctor" << std::endl;

  quality.reset(new SiStripQuality());
}


boost::shared_ptr<SiStripQuality> SiStripQualityESProducer::produce(const SiStripQualityRcd& iRecord)
{
  
  edm::LogInfo("SiStripQualityESProducer") << "produce called" << std::endl;

  quality->clear();

  edm::ESHandle<SiStripBadStrip> obj;
  edm::ESHandle<SiStripDetCabling> cabling;
  edm::ESHandle<SiStripDetVOff> Voff;
  edm::ESHandle<RunInfo> runInfo;

  std::string tagName;  
  std::string recordName;

  bool doRunInfo = false;
  std::string runInfoTagName = "";

  // Set the debug output level
  quality->setPrintDebugOutput( pset_.getParameter<bool>("PrintDebugOutput") );
  // Set the protection against empty RunInfo objects
  quality->setUseEmptyRunInfo( pset_.getParameter<bool>("UseEmptyRunInfo") );

  for( Parameters::iterator itToGet = toGet.begin(); itToGet != toGet.end(); ++itToGet ) {
    tagName = itToGet->getParameter<std::string>("tag");
    recordName = itToGet->getParameter<std::string>("record");

    edm::LogInfo("SiStripQualityESProducer") << "[SiStripQualityESProducer::produce] Getting data from record " << recordName << " with tag " << tagName << std::endl;

    if (recordName=="SiStripBadModuleRcd"){
      iRecord.getRecord<SiStripBadModuleRcd>().get(tagName,obj); 
      quality->add( obj.product() );
    } else if (recordName=="SiStripBadFiberRcd"){
      iRecord.getRecord<SiStripBadFiberRcd>().get(tagName,obj); 
      quality->add( obj.product() );    
    } else if (recordName=="SiStripBadChannelRcd"){
      iRecord.getRecord<SiStripBadChannelRcd>().get(tagName,obj);
      quality->add( obj.product() );    
    } else if (recordName=="SiStripBadStripRcd"){
      iRecord.getRecord<SiStripBadStripRcd>().get(tagName,obj); 
      quality->add( obj.product() );    
    } else if (recordName=="SiStripDetCablingRcd"){
      iRecord.getRecord<SiStripDetCablingRcd>().get(tagName,cabling);
      quality->add( cabling.product() );
    } else if (recordName=="SiStripDetVOffRcd"){
      iRecord.getRecord<SiStripDetVOffRcd>().get(tagName,Voff);
      quality->add( Voff.product() );
    } else if (recordName=="RunInfoRcd") {
      runInfoTagName = tagName;
      doRunInfo = true;
    } else {
      edm::LogError("SiStripQualityESProducer") << "[SiStripQualityESProducer::produce] Skipping the requested data for unexisting record " << recordName << " with tag " << tagName << std::endl;
      continue;
    }
  }
  // We do this after all the others so we know it is done after the DetCabling (if any)
  if( doRunInfo ) {
    iRecord.getRecord<RunInfoRcd>().get(runInfoTagName,runInfo);
    quality->add( runInfo.product() );
  }

  quality->cleanUp();

  if(pset_.getParameter<bool>("ReduceGranularity")){
      quality->ReduceGranularity(pset_.getParameter<double>("ThresholdForReducedGranularity"));
      quality->cleanUp(true);
  }

  quality->fillBadComponents();
  
  return quality;
}

