#include "DQM/SiPixelHistoricInfoClient/plugins/SiPixelHistoryDQMService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Diagnostic/interface/HDQMfitUtilities.h"
#include "DQM/SiPixelHistoricInfoClient/interface/SiPixelSummary.h"


SiPixelHistoryDQMService::SiPixelHistoryDQMService(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg)
: DQMHistoryServiceBase::DQMHistoryServiceBase(iConfig, aReg), iConfig_(iConfig)
{
  edm::LogInfo("SiPixelHistoryDQMService") <<  "[SiPixelHistoryDQMService::SiPixelHistoryDQMService]";
}


SiPixelHistoryDQMService::~SiPixelHistoryDQMService() { 
  edm::LogInfo("SiPixelHistoryDQMService") <<  "[SiPixelHistoryDQMService::~SiPixelHistoryDQMService]";
}


uint32_t SiPixelHistoryDQMService::returnDetComponent(const MonitorElement* ME){
  LogTrace("SiPixelHistoryDQMService") <<  "[SiPixelHistoryDQMService::returnDetComponent]";
  std::string str=ME->getName();
  size_t __key_length__=7;
  size_t __detid_length__=9;


  if(str.find("__det__")!= std::string::npos){
    return atoi(str.substr(str.find("__det__")+__key_length__,__detid_length__).c_str());
  }
  else if(str.find("Barrel")!= std::string::npos)
  {return sipixelsummary::Barrel;}
  else if(str.find("Shell_mI")!= std::string::npos)
  {return sipixelsummary::Shell_mI;}
  else if(str.find("Shell_mO")!= std::string::npos)
  {return sipixelsummary::Shell_mO;}
  else if(str.find("Shell_pI")!= std::string::npos)
  {return sipixelsummary::Shell_pI;}
  else if(str.find("Shell_pO")!= std::string::npos)
  {return sipixelsummary::Shell_pO;}
  else if(str.find("Endcap")!= std::string::npos)
  {return sipixelsummary::Endcap;}
  else if(str.find("HalfCylinder_mI")!= std::string::npos)
  {return sipixelsummary::HalfCylinder_mI;}
  else if(str.find("HalfCylinder_mO")!= std::string::npos)
  {return sipixelsummary::HalfCylinder_mO;}
  else if(str.find("HalfCylinder_pI")!= std::string::npos)
  {return sipixelsummary::HalfCylinder_pI;}
  else if(str.find("HalfCylinder_pO")!= std::string::npos)
  {return sipixelsummary::HalfCylinder_pO;}
  else {
    return sipixelsummary::TRACKER; //Full Tracker
  }

}





//Example on how to define an user function for the statistic extraction
bool SiPixelHistoryDQMService::setDBLabelsForUser  (std::string& keyName, std::vector<std::string>& userDBContent){
  userDBContent.push_back(keyName+std::string("@")+std::string("entries"));
  userDBContent.push_back(keyName+std::string("@")+std::string("ymean"));
  return true;
}
bool SiPixelHistoryDQMService::setDBValuesForUser(std::vector<MonitorElement*>::const_iterator iterMes, HDQMSummary::InputVector& values  ){
  TH1F* Hist = (TH1F*) (*iterMes)->getTH1F()->Clone();
  Hist->Fit("pol0");
  TF1* Fit = Hist->GetFunction("pol0");
  float FitValue = Fit ? Fit->GetParameter(0) : 0;

  values.push_back( (*iterMes)->getEntries() );
  values.push_back( FitValue );
  return true;
}

