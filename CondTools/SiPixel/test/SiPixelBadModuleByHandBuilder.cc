#include "SiPixelBadModuleByHandBuilder.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"

#include <cmath>
#include <fstream>
#include <iostream>

SiPixelBadModuleByHandBuilder::SiPixelBadModuleByHandBuilder(const edm::ParameterSet& iConfig)
    : ConditionDBWriter<SiPixelQuality>(iConfig) {
  printdebug_ = iConfig.getUntrackedParameter<bool>("printDebug", false);
  BadModuleList_ = iConfig.getUntrackedParameter<Parameters>("BadModuleList");
  ROCListFile_ = iConfig.getUntrackedParameter<std::string>("ROCListFile");
}

SiPixelBadModuleByHandBuilder::~SiPixelBadModuleByHandBuilder() {}

std::unique_ptr<SiPixelQuality> SiPixelBadModuleByHandBuilder::getNewObject() {
  auto obj = std::make_unique<SiPixelQuality>();

  for (Parameters::iterator it = BadModuleList_.begin(); it != BadModuleList_.end(); ++it) {
    if (printdebug_) {
      edm::LogInfo("SiPixelBadModuleByHandBuilder") << " BadModule " << *it << " \t" << std::endl;
    }

    SiPixelQuality::disabledModuleType BadModule;
    BadModule.errorType = 3;
    BadModule.BadRocs = 0;
    BadModule.DetID = it->getParameter<uint32_t>("detid");
    std::string errorstring = it->getParameter<std::string>("errortype");
    if (printdebug_) {
      edm::LogInfo("SiPixelBadModuleByHandBuilder")
          << "now looking at detid " << BadModule.DetID << ", string " << errorstring << std::endl;
    }

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

    if (errorstring == "whole") {
      BadModule.errorType = 0;
      BadModule.BadRocs = 65535;
    }  //corresponds to all rocs being bad
    else if (errorstring == "tbmA") {
      BadModule.errorType = 1;
      BadModule.BadRocs = 255;
    }  //corresponds to Rocs 0-7 being bad
    else if (errorstring == "tbmB") {
      BadModule.errorType = 2;
      BadModule.BadRocs = 65280;
    }  //corresponds to Rocs 8-15 being bad
    else if (errorstring == "none") {
      BadModule.errorType = 3;
      //       badroclist_ = iConfig.getUntrackedParameter<std::vector<uint32_t> >("badroclist");
      std::vector<uint32_t> BadRocList = it->getParameter<std::vector<uint32_t> >("badroclist");
      short badrocs = 0;
      for (std::vector<uint32_t>::iterator iter = BadRocList.begin(); iter != BadRocList.end(); ++iter) {
        badrocs += 1 << *iter;  // 1 << *iter = 2^{*iter} using bitwise shift
      }
      BadModule.BadRocs = badrocs;
    }

    else
      edm::LogError("SiPixelQuality") << "trying to fill error type " << errorstring << ", which is not defined!";
    obj->addDisabledModule(BadModule);
  }

  // fill DB from DQM list
  if (!ROCListFile_.empty()) {
    std::map<uint32_t, uint32_t> disabledModules;
    std::ifstream aFile(ROCListFile_.c_str());
    std::string aLine;
    while (std::getline(aFile, aLine)) {
      char name[100];
      int roc;
      sscanf(aLine.c_str(), "%s %d", name, &roc);
      uint32_t detId;
      if (name[0] == 'B') {
        PixelBarrelName bn(name, true);
        detId = bn.getDetId(tTopo_);
      } else {
        PixelEndcapName en(name, true);
        detId = en.getDetId(tTopo_);
      }
      std::map<uint32_t, uint32_t>::iterator it = disabledModules.find(detId);
      if (it == disabledModules.end())
        it = disabledModules.insert(disabledModules.begin(), std::make_pair(detId, 0));
      it->second |= 1 << roc;
      //std::cout<<"New module read "<<name<<" "<<roc<<" --> "<<detId<<" "<<std::bitset<32>(it->second)<<std::endl;
    }

    for (const auto& it : disabledModules) {
      SiPixelQuality::disabledModuleType BadModule;
      BadModule.DetID = it.first;
      if (it.second == 65535) {  // "whole"
        BadModule.errorType = 0;
        BadModule.BadRocs = 65535;
      }                             //corresponds to all rocs being bad
      else if (it.second == 255) {  // "tbmA"
        BadModule.errorType = 1;
        BadModule.BadRocs = 255;
      }                               //corresponds to Rocs 0-7 being bad
      else if (it.second == 65280) {  // "tbmB"
        BadModule.errorType = 2;
        BadModule.BadRocs = 65280;
      }       //corresponds to Rocs 8-15 being bad
      else {  // "none"
        BadModule.errorType = 3;
        BadModule.BadRocs = it.second;
      }

      obj->addDisabledModule(BadModule);
      if (printdebug_) {
        edm::LogVerbatim("SiPixelBadModuleByHandBuilder")
            << "New module added: " << tTopo_->print(BadModule.DetID) << ", errorType: " << BadModule.errorType
            << ", BadRocs: " << std::bitset<16>(it.second) << std::endl;
      }
    }
  }

  return obj;
}
