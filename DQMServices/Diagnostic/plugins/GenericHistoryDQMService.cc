#include "DQMServices/Diagnostic/plugins/GenericHistoryDQMService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorElement.h"


GenericHistoryDQMService::GenericHistoryDQMService(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg)
: DQMHistoryServiceBase::DQMHistoryServiceBase(iConfig, aReg), iConfig_(iConfig)
{
  edm::LogInfo("GenericHistoryDQMService") <<  "[GenericHistoryDQMService::GenericHistoryDQMService]";
}


GenericHistoryDQMService::~GenericHistoryDQMService() { 
  edm::LogInfo("GenericHistoryDQMService") <<  "[GenericHistoryDQMService::~GenericHistoryDQMService]";
}


uint32_t GenericHistoryDQMService::returnDetComponent(const MonitorElement* ME){
  LogTrace("GenericHistoryDQMService") <<  "[GenericHistoryDQMService::returnDetComponent] returning value defined in the configuration Pset \"DetectorId\"";
  return iConfig_.getParameter<uint32_t>("DetectorId");
}

//Example on how to define an user function for the statistic extraction
bool GenericHistoryDQMService::setDBLabelsForUser  (std::string& keyName, std::vector<std::string>& userDBContent){
  userDBContent.push_back(keyName+std::string("@")+std::string("userExample_XMax"));
  userDBContent.push_back(keyName+std::string("@")+std::string("userExample_mean"));
  return true;
}
bool GenericHistoryDQMService::setDBValuesForUser(std::vector<MonitorElement*>::const_iterator iterMes, HDQMSummary::InputVector& values  ){
  values.push_back( (*iterMes)->getTH1F()->GetXaxis()->GetBinCenter((*iterMes)->getTH1F()->GetMaximumBin()));
  values.push_back( (*iterMes)->getMean() );
  return true;
}



