#include "DQM/SiStripHistoricInfoClient/plugins/SiStripNewHistoricDQMService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Diagnostic/interface/HDQMfitUtilities.h"

/*#include <string>
#include <sstream>
#include <cctype>
#include <time.h>
#include <boost/cstdint.hpp>*/


SiStripNewHistoricDQMService::SiStripNewHistoricDQMService(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg)
: DQMHistoryServiceBase::DQMHistoryServiceBase(iConfig, aReg), iConfig_(iConfig)
{
  edm::LogInfo("SiStripNewHistoricDQMService") <<  "[SiStripNewHistoricDQMService::SiStripNewHistoricDQMService]";
}


SiStripNewHistoricDQMService::~SiStripNewHistoricDQMService() { 
  edm::LogInfo("SiStripNewHistoricDQMService") <<  "[SiStripNewHistoricDQMService::~SiStripNewHistoricDQMService]";
}


uint32_t SiStripNewHistoricDQMService::returnDetComponent(std::string& str){
  LogTrace("SiStripNewHistoricDQMService") <<  "[SiStripNewHistoricDQMService::returnDetComponent]";

  size_t __key_length__=7;
  size_t __detid_length__=9;

  if(str.find("__det__")!= std::string::npos){
    return atoi(str.substr(str.find("__det__")+__key_length__,__detid_length__).c_str());
  }
  //TIB
  else if(str.find("TIB")!= std::string::npos){
    if (str.find("layer")!= std::string::npos) 
      return hdqmsummary::TIB*10
	+atoi(str.substr(str.find("layer__")+__key_length__,1).c_str()); 
    return hdqmsummary::TIB;
  }
  //TOB
  else if(str.find("TOB")!= std::string::npos){
    if (str.find("layer")!= std::string::npos) 
      return hdqmsummary::TOB*10
	+atoi(str.substr(str.find("layer__")+__key_length__,1).c_str());
    return hdqmsummary::TOB;
  }
  //TID
  else if(str.find("TID")!= std::string::npos){  
    if (str.find("side")!= std::string::npos){
      if (str.find("wheel")!= std::string::npos){
	return hdqmsummary::TID*100
	  +atoi(str.substr(str.find("_side__")+__key_length__,1).c_str())*10
	  +atoi(str.substr(str.find("wheel__")+__key_length__,1).c_str());
      }
      return hdqmsummary::TID*10
	+atoi(str.substr(str.find("_side__")+__key_length__,1).c_str());
    }
    return hdqmsummary::TID;
  } 
  //TEC
  else if(str.find("TEC")!= std::string::npos){  
    if (str.find("side")!= std::string::npos){
      if (str.find("wheel")!= std::string::npos){
	return hdqmsummary::TEC*100
	  +atoi(str.substr(str.find("_side__")+__key_length__,1).c_str())*10
	  +atoi(str.substr(str.find("wheel__")+__key_length__,1).c_str());
      }
      return hdqmsummary::TEC*10
	+atoi(str.substr(str.find("_side__")+__key_length__,1).c_str());
    }
    return hdqmsummary::TEC;
  } 
  else 
    return hdqmsummary::TRACKER; //Full Tracker
}



