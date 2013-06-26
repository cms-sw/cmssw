#ifndef DQMOFFLINE_TRIGGER_EGHLTCUTMASKS
#define DQMOFFLINE_TRIGGER_EGHLTCUTMASKS

//this is a bunch of usefull cut masks to turn off / on cuts
//this class may disappear in the future
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMOffline/Trigger/interface/EgHLTEgCutCodes.h"



namespace egHLT {
  
  struct CutMasks {
    int stdEle;
    int tagEle;
    int probeEle;
    int fakeEle;
    int trigTPEle;
    int trigTPPho;
    int stdPho;

    void setup(const edm::ParameterSet& conf){
      stdEle = EgCutCodes::getCode(conf.getParameter<std::string>("stdEle"));
      tagEle = EgCutCodes::getCode(conf.getParameter<std::string>("tagEle"));
      probeEle= EgCutCodes::getCode(conf.getParameter<std::string>("probeEle"));
      fakeEle = EgCutCodes::getCode(conf.getParameter<std::string>("fakeEle"));
      trigTPEle = EgCutCodes::getCode(conf.getParameter<std::string>("trigTPEle"));
      trigTPPho = EgCutCodes::getCode(conf.getParameter<std::string>("trigTPPho"));
      stdPho = EgCutCodes::getCode(conf.getParameter<std::string>("stdPho"));
    }
  };

  

}

#endif
