#include "CalibTracker/SiStripQuality/plugins/SiStripBadModuleByHandBuilder.h"


#include <iostream>
#include <fstream>


SiStripBadModuleByHandBuilder::SiStripBadModuleByHandBuilder(const edm::ParameterSet& iConfig) : ConditionDBWriter<SiStripBadStrip>::ConditionDBWriter<SiStripBadStrip>(iConfig){

  edm::LogInfo("SiStripBadModuleByHandBuilder") << " ctor ";
  fp_ = iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"));
  BadModuleList_ = iConfig.getUntrackedParameter<std::vector<uint32_t> >("BadModuleList");
  printdebug_ = iConfig.getUntrackedParameter<bool>("printDebug",false);

  reader = new SiStripDetInfoFileReader(fp_.fullPath());  
}


SiStripBadModuleByHandBuilder::~SiStripBadModuleByHandBuilder(){
  edm::LogInfo("SiStripBadModuleByHandBuilder") << " dtor";
}

SiStripBadStrip* SiStripBadModuleByHandBuilder::getNewObject(){
  
  edm::LogInfo("SiStripBadModuleByHandBuilder") <<"SiStripBadModuleByHandBuilder::getNewObject called"<<std::endl;
  
  SiStripBadStrip* obj = new SiStripBadStrip();

  unsigned int firstBadStrip=0;
  unsigned short NconsecutiveBadStrips;
  unsigned int theBadStripRange; 

  for(std::vector<uint32_t>::const_iterator it=BadModuleList_.begin(); it!=BadModuleList_.end(); ++it){
    
    std::vector<unsigned int> theSiStripVector;
    
    NconsecutiveBadStrips=reader->getNumberOfApvsAndStripLength(*it).first*128;
    theBadStripRange = obj->encode(firstBadStrip,NconsecutiveBadStrips);
    if (printdebug_)
      edm::LogInfo("SiStripBadModuleByHandBuilder") << " BadModule " << *it << " \t"
						   << " firstBadStrip " << firstBadStrip << "\t "
						   << " NconsecutiveBadStrips " << NconsecutiveBadStrips << "\t "
						   << " packed integer " << std::hex << theBadStripRange  << std::dec
						   << std::endl; 	    
    
    theSiStripVector.push_back(theBadStripRange);
    SiStripBadStrip::Range range(theSiStripVector.begin(),theSiStripVector.end());
    if ( ! obj->put(*it,range) )
      edm::LogError("SiStripBadModuleByHandBuilder")<<"[SiStripBadModuleByHandBuilder::analyze] detid already exists"<<std::endl;
  }
  return obj;
}


