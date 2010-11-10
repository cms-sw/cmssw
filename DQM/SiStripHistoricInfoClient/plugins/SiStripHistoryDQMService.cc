#include "DQM/SiStripHistoricInfoClient/plugins/SiStripHistoryDQMService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Diagnostic/interface/HDQMfitUtilities.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"


SiStripHistoryDQMService::SiStripHistoryDQMService(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg)
: DQMHistoryServiceBase::DQMHistoryServiceBase(iConfig, aReg), iConfig_(iConfig)
{
  edm::LogInfo("SiStripHistoryDQMService") <<  "[SiStripHistoryDQMService::SiStripHistoryDQMService]";
}


SiStripHistoryDQMService::~SiStripHistoryDQMService() { 
  edm::LogInfo("SiStripHistoryDQMService") <<  "[SiStripHistoryDQMService::~SiStripHistoryDQMService]";
}


uint32_t SiStripHistoryDQMService::returnDetComponent(const MonitorElement* ME){
  LogTrace("SiStripHistoryDQMService") <<  "[SiStripHistoryDQMService::returnDetComponent]";
  std::string str=ME->getName();
  size_t __key_length__=7;
  size_t __detid_length__=9;

  uint32_t layer=0,side=0;

  if(str.find("__det__")!= std::string::npos){
    return atoi(str.substr(str.find("__det__")+__key_length__,__detid_length__).c_str());
  }
  //TIB
  else if(str.find("TIB")!= std::string::npos){
    if (str.find("layer")!= std::string::npos) 
      layer=atoi(str.substr(str.find("layer__")+__key_length__,1).c_str());
    return TIBDetId(layer,0,0,0,0,0).rawId();
  }
  //TOB
  else if(str.find("TOB")!= std::string::npos){
    if (str.find("layer")!= std::string::npos) 
      layer=atoi(str.substr(str.find("layer__")+__key_length__,1).c_str());
    return TOBDetId(layer,0,0,0,0).rawId();
  }
  //TID
  else if(str.find("TID")!= std::string::npos){  
    if (str.find("side")!= std::string::npos){
      side=atoi(str.substr(str.find("_side__")+__key_length__,1).c_str());
      if (str.find("wheel")!= std::string::npos){
	layer=atoi(str.substr(str.find("wheel__")+__key_length__,1).c_str());
      }
    }
    return TIDDetId(side,layer,0,0,0,0).rawId();
  } 
  //TEC
  else if(str.find("TEC")!= std::string::npos){  
    if (str.find("side")!= std::string::npos){
      layer=atoi(str.substr(str.find("wheel__")+__key_length__,1).c_str());
      if (str.find("wheel")!= std::string::npos){
	side=atoi(str.substr(str.find("_side__")+__key_length__,1).c_str());
      }
    }
    return TECDetId(side,layer,0,0,0,0,0).rawId();
  } 
  else 
    return SiStripDetId(DetId::Tracker,0).rawId(); //Full Tracker
}

//Example on how to define an user function for the statistic extraction
bool SiStripHistoryDQMService::setDBLabelsForUser  (std::string& keyName, std::vector<std::string>& userDBContent){
  userDBContent.push_back(keyName+std::string("@")+std::string("userExample_XMax"));
  userDBContent.push_back(keyName+std::string("@")+std::string("userExample_mean"));
  return true;
}
bool SiStripHistoryDQMService::setDBValuesForUser(std::vector<MonitorElement*>::const_iterator iterMes, HDQMSummary::InputVector& values  ){
  values.push_back( (*iterMes)->getTH1F()->GetXaxis()->GetBinCenter((*iterMes)->getTH1F()->GetMaximumBin()));
  values.push_back( (*iterMes)->getMean() );
  return true;
}

