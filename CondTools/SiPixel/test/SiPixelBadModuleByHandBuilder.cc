#include "CondTools/SiPixel/test/SiPixelBadModuleByHandBuilder.h"

#include <math.h>
#include <iostream>
#include <fstream>


SiPixelBadModuleByHandBuilder::SiPixelBadModuleByHandBuilder(const edm::ParameterSet& iConfig) : ConditionDBWriter<SiPixelQuality>(iConfig){

  printdebug_ = iConfig.getUntrackedParameter<bool>("printDebug",false);
  BadModuleList_ = iConfig.getUntrackedParameter<Parameters>("BadModuleList");
 
}


SiPixelBadModuleByHandBuilder::~SiPixelBadModuleByHandBuilder(){
}

std::unique_ptr<SiPixelQuality> SiPixelBadModuleByHandBuilder::getNewObject(){
  
  auto obj = std::make_unique<SiPixelQuality>();

  for(Parameters::iterator it = BadModuleList_.begin(); it != BadModuleList_.end(); ++it) {

    if (printdebug_)
     edm::LogInfo("SiPixelBadModuleByHandBuilder") << " BadModule " << *it << " \t" << std::endl; 	    


    SiPixelQuality::disabledModuleType BadModule;
    BadModule.errorType = 3; BadModule.BadRocs = 0;
    BadModule.DetID = it->getParameter<uint32_t>("detid");
    std::string errorstring = it->getParameter<std::string>("errortype");
    std::cout << "now looking at detid " << BadModule.DetID << ", string " << errorstring << std::endl;
 
      //////////////////////////////////////
      //  errortype "whole" = int 0 in DB //
      //  errortype "tbmA" = int 1 in DB  //
      //  errortype "tbmB" = int 2 in DB  //
      //  errortype "none" = int 3 in DB  //
      //////////////////////////////////////
    
      /////////////////////////////////////////////////
      //each bad roc correspond to a bit to 1: num=  //
      // 0 <-> all good rocs                         //
      // 1 <-> only roc 0 bad                        //
      // 2<-> only roc 1 bad                         //
      // 3<->  roc 0 and 1 bad                       //
      // 4 <-> only roc 2 bad                        //
      //  ...                                        //
      /////////////////////////////////////////////////



      if(errorstring=="whole"){
       BadModule.errorType = 0;
       BadModule.BadRocs = 65535;} //corresponds to all rocs being bad
      else if(errorstring=="tbmA"){
       BadModule.errorType = 1;
       BadModule.BadRocs = 255;}  //corresponds to Rocs 0-7 being bad
      else if(errorstring=="tbmB"){
       BadModule.errorType = 2;
       BadModule.BadRocs = 65280;} //corresponds to Rocs 8-15 being bad
      else if(errorstring=="none"){
	BadModule.errorType = 3;
	//       badroclist_ = iConfig.getUntrackedParameter<std::vector<uint32_t> >("badroclist");
	std::vector<uint32_t> BadRocList = it->getParameter<std::vector<uint32_t> >("badroclist");
        short badrocs = 0;
        for(std::vector<uint32_t>::iterator iter = BadRocList.begin(); iter != BadRocList.end(); ++iter){
	  badrocs +=  1 << *iter; // 1 << *iter = 2^{*iter} using bitwise shift 
	}
        BadModule.BadRocs = badrocs;
      }
        


    else
      edm::LogError("SiPixelQuality") << "trying to fill error type " << errorstring << ", which is not defined!" ;
     obj->addDisabledModule(BadModule);

    }
  return obj;
}


