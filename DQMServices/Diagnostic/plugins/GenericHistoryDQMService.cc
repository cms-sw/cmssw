#include "DQMServices/Diagnostic/plugins/GenericHistoryDQMService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorElement.h"


GenericHistoryDQMService::GenericHistoryDQMService(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg)
: DQMHistoryServiceBase::DQMHistoryServiceBase(iConfig, aReg), iConfig_(iConfig)
{
  edm::LogInfo("GenericHistoryDQMService") <<  "[GenericHistoryDQMService::GenericHistoryDQMService]";
}

GenericHistoryDQMService::~GenericHistoryDQMService()
{
  edm::LogInfo("GenericHistoryDQMService") <<  "[GenericHistoryDQMService::~GenericHistoryDQMService]";
}

uint32_t GenericHistoryDQMService::returnDetComponent(const MonitorElement* ME)
{
  LogTrace("GenericHistoryDQMService") <<  "[GenericHistoryDQMService::returnDetComponent] returning value defined in the configuration Pset \"DetectorId\"";
  return iConfig_.getParameter<uint32_t>("DetectorId");
}

/// Example on how to define an user function for the statistic extraction
bool GenericHistoryDQMService::setDBLabelsForUser  (std::string& keyName, std::vector<std::string>& userDBContent, std::string& quantity )
{
  if(quantity=="userExample_XMax"){
    userDBContent.push_back(keyName+std::string("@")+std::string("userExample_XMax"));
  }
  else if(quantity=="userExample_mean"){
      userDBContent.push_back(keyName+std::string("@")+std::string("userExample_mean"));
  }
  else{
    edm::LogError("DQMHistoryServiceBase") 
      << "Quantity " << quantity
      << " cannot be handled\nAllowed quantities are" 
      << "\n  'stat'   that includes: entries, mean, rms"
      << "\n  'landau' that includes: landauPeak, landauPeakErr, landauSFWHM, landauChi2NDF"
      << "\n  'gauss'  that includes: gaussMean, gaussSigma, gaussChi2NDF"
      << "\n or a specific user quantity that should be implemented in the user functions GenericHistoryDQMService::setDBLabelsForUser"
      << std::endl;
    return false;
  }
  return true;
}

bool GenericHistoryDQMService::setDBValuesForUser(std::vector<MonitorElement*>::const_iterator iterMes, HDQMSummary::InputVector& values, std::string& quantity )
{
  if(quantity=="userExample_XMax"){
    values.push_back( (*iterMes)->getTH1F()->GetXaxis()->GetBinCenter((*iterMes)->getTH1F()->GetMaximumBin()));
  }
  else if(quantity=="userExample_mean"){
    values.push_back( (*iterMes)->getMean() );
  }
  else{
    return false;
  }
  return true;
}



