#include "CondTools/SiStrip/plugins/SiStripBadFiberBuilder.h"

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

#include <iostream>
#include <fstream>
#include <sstream>

SiStripBadFiberBuilder::SiStripBadFiberBuilder(const edm::ParameterSet& iConfig) : ConditionDBWriter<SiStripBadStrip>(iConfig){

  fp_ = iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"));
  printdebug_ = iConfig.getUntrackedParameter<bool>("printDebug",false);
  BadComponentList_ =  iConfig.getUntrackedParameter<Parameters>("BadComponentList");
}


SiStripBadFiberBuilder::~SiStripBadFiberBuilder(){
}

std::unique_ptr<SiStripBadStrip> SiStripBadFiberBuilder::getNewObject() {
  
  edm::LogInfo("SiStripBadFiberBuilder") << "... creating dummy SiStripBadStrip Data" << std::endl;
  
  auto obj =  std::make_unique<SiStripBadStrip>();

  SiStripDetInfoFileReader reader(fp_.fullPath());
  
  std::stringstream ss;
  for(Parameters::iterator iBadComponent = BadComponentList_.begin(); iBadComponent != BadComponentList_.end(); ++iBadComponent ) {
    
    uint32_t BadModule_ = iBadComponent->getParameter<uint32_t>("BadModule");
    std::vector<uint32_t> BadApvList_ = iBadComponent->getParameter<std::vector<uint32_t> >("BadApvList");

    std::vector<unsigned int> theSiStripVector;
    
    unsigned short firstBadStrip=0, NconsecutiveBadStrips=0;
    unsigned int theBadStripRange;
    
    for(std::vector<uint32_t>::const_iterator is=BadApvList_.begin(); is!=BadApvList_.end(); ++is){

      firstBadStrip=(*is)*128;
      NconsecutiveBadStrips=128;
      
      theBadStripRange = obj->encode(firstBadStrip,NconsecutiveBadStrips);
      
      if (printdebug_)
	ss << "detid " << BadModule_ << " \t"
	   << " firstBadStrip " << firstBadStrip << "\t "
	   << " NconsecutiveBadStrips " << NconsecutiveBadStrips << "\t "
	   << " packed integer " << std::hex << theBadStripRange  << std::dec
	   << std::endl; 	    
      
      theSiStripVector.push_back(theBadStripRange);
    }      
     
    SiStripBadStrip::Range range(theSiStripVector.begin(),theSiStripVector.end());
    if ( ! obj->put(BadModule_,range) )
      edm::LogError("SiStripBadFiberBuilder")<<"[SiStripBadFiberBuilder::analyze] detid already exists"<<std::endl;
  }
  if (printdebug_)
    edm::LogInfo("SiStripBadFiberBuilder") << ss.str();

  return obj;
}


