#include "CalibTracker/SiStripQuality/plugins/SiStripQualityHotStripIdentifier.h"

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

#include <iostream>
#include <fstream>


SiStripQualityHotStripIdentifier::SiStripQualityHotStripIdentifier(const edm::ParameterSet& iConfig) : ConditionDBWriter<SiStripBadStrip>::ConditionDBWriter<SiStripBadStrip>(iConfig){

  edm::LogInfo("SiStripQualityHotStripIdentifier") << " ctor ";
  fp_ = iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"));
  BadModuleList_ = iConfig.getUntrackedParameter<std::vector<uint32_t> >("BadModuleList");
  printdebug_ = iConfig.getUntrackedParameter<bool>("printDebug",false);

  reader = new SiStripDetInfoFileReader(fp_.fullPath());  
}


SiStripQualityHotStripIdentifier::~SiStripQualityHotStripIdentifier(){
  edm::LogInfo("SiStripQualityHotStripIdentifier") << " dtor";
}

SiStripBadStrip* SiStripQualityHotStripIdentifier::getNewObject(){

  edm::LogInfo("SiStripQualityHotStripIdentifier") <<"SiStripQualityHotStripIdentifier::getNewObject called"<<std::endl;
  
  SiStripBadStrip* obj = new SiStripBadStrip();

  /*  
  unsigned int firstBadStrip=0;
  unsigned short NconsecutiveBadStrips;
  unsigned int theBadStripRange; 

  for(std::vector<uint32_t>::const_iterator it=BadModuleList_.begin(); it!=BadModuleList_.end(); ++it){
    
    std::vector<unsigned int> theSiStripVector;
    
    NconsecutiveBadStrips=reader->getNumberOfApvsAndStripLength(*it).first*128;
    theBadStripRange = obj->encode(firstBadStrip,NconsecutiveBadStrips);
    if (printdebug_)
      edm::LogInfo("SiStripQualityHotStripIdentifier") << " BadModule " << *it << " \t"
						   << " firstBadStrip " << firstBadStrip << "\t "
						   << " NconsecutiveBadStrips " << NconsecutiveBadStrips << "\t "
						   << " packed integer " << std::hex << theBadStripRange  << std::dec
						   << std::endl; 	    
    
    theSiStripVector.push_back(theBadStripRange);
    SiStripBadStrip::Range range(theSiStripVector.begin(),theSiStripVector.end());
    if ( ! obj->put(*it,range) )
      edm::LogError("SiStripQualityHotStripIdentifier")<<"[SiStripQualityHotStripIdentifier::analyze] detid already exists"<<std::endl;
  }
  */
  return obj;
}


