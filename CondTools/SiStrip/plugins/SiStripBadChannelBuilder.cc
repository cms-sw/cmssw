#include "CondTools/SiStrip/plugins/SiStripBadChannelBuilder.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

#include <iostream>
#include <fstream>


SiStripBadChannelBuilder::SiStripBadChannelBuilder(const edm::ParameterSet& iConfig) : ConditionDBWriter<SiStripBadStrip>(iConfig){

  fp_ = iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"));
  printdebug_ = iConfig.getUntrackedParameter<bool>("printDebug",false);
  BadComponentList_ =  iConfig.getUntrackedParameter<Parameters>("BadComponentList");
}


SiStripBadChannelBuilder::~SiStripBadChannelBuilder(){
}

void SiStripBadChannelBuilder::algoAnalyze(const edm::Event & event, const edm::EventSetup& iSetup){
  
  unsigned int run=event.id().run();

  edm::LogInfo("SiStripBadChannelBuilder") << "... creating dummy SiStripBadStrip Data for Run " << run << "\n " << std::endl;
  
  SiStripBadStrip* obj = new SiStripBadStrip();

  SiStripDetInfoFileReader reader(fp_.fullPath());
  
  const std::vector<uint32_t> DetIds = reader.getAllDetIds();
  
  for(Parameters::iterator iBadComponent = BadComponentList_.begin(); iBadComponent != BadComponentList_.end(); ++iBadComponent ) {
    
    uint32_t BadModule_ = iBadComponent->getParameter<uint32_t>("BadModule");
    std::vector<uint32_t> BadChannelList_ = iBadComponent->getParameter<std::vector<uint32_t> >("BadChannelList");

    std::vector<unsigned int> theSiStripVector;
    unsigned int NStrips=reader.getNumberOfApvsAndStripLength(BadModule_).first*128;   
    
    uint32_t lastBad=999;
    unsigned short firstBadStrip=0, NconsecutiveBadStrips=0;
    unsigned int theBadStripRange;

    for(std::vector<uint32_t>::const_iterator is=BadChannelList_.begin(); is!=BadChannelList_.end(); ++is){
      if (*is>NStrips-1)
	break;
      if (*is!=lastBad+1){
	//new set 

	if ( lastBad!=999 ){
	  //save previous set
	  theBadStripRange = obj->encode(firstBadStrip,NconsecutiveBadStrips);

	  if (printdebug_)
	    edm::LogInfo("SiStripBadChannelBuilder") << "detid " << BadModule_ << " \t"
						   << " firstBadStrip " << firstBadStrip << "\t "
						   << " NconsecutiveBadStrips " << NconsecutiveBadStrips << "\t "
						   << " packed integer " << std::hex << theBadStripRange  << std::dec
						   << std::endl; 	    
	  
	  theSiStripVector.push_back(theBadStripRange);
	}
	
	firstBadStrip=*is;
	NconsecutiveBadStrips=0;
      } 	
      NconsecutiveBadStrips++;
      lastBad=*is;
    }

    theBadStripRange = obj->encode(firstBadStrip,NconsecutiveBadStrips);
    if (printdebug_)
      edm::LogInfo("SiStripBadChannelBuilder") << "detid " << BadModule_ << " \t"
					     << " firstBadStrip " << firstBadStrip << "\t "
					     << " NconsecutiveBadStrips " << NconsecutiveBadStrips << "\t "
					     << " packed integer " << std::hex << theBadStripRange  << std::dec
					     << std::endl; 	    
	  
    theSiStripVector.push_back(theBadStripRange);
        
    SiStripBadStrip::Range range(theSiStripVector.begin(),theSiStripVector.end());
    if ( ! obj->put(BadModule_,range) )
    edm::LogError("SiStripBadChannelBuilder")<<"[SiStripBadChannelBuilder::analyze] detid already exists"<<std::endl;
  }
 //End now write sistripbadChannel data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if( mydbservice.isAvailable() ){
    if ( mydbservice->isNewTagRequest("SiStripBadStripRcd") ){
      mydbservice->createNewIOV<SiStripBadStrip>(obj,mydbservice->beginOfTime(),mydbservice->endOfTime(),"SiStripBadStripRcd");
    } else {
      //mydbservice->createNewIOV<SiStripBadStrip>(obj,mydbservice->currentTime(),"SiStripBadStripRcd");
      mydbservice->appendSinceTime<SiStripBadStrip>(obj,mydbservice->currentTime(),"SiStripBadStripRcd");
    }
  }else{
    edm::LogError("SiStripBadStripBuilder")<<"Service is unavailable"<<std::endl;
  }

}


