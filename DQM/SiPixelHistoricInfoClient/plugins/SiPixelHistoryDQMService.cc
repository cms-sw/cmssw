#include "DQM/SiPixelHistoricInfoClient/plugins/SiPixelHistoryDQMService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Diagnostic/interface/HDQMfitUtilities.h"
#include "DQM/SiPixelHistoricInfoClient/interface/SiPixelSummary.h"


SiPixelHistoryDQMService::SiPixelHistoryDQMService(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg)
: DQMHistoryServiceBase::DQMHistoryServiceBase(iConfig, aReg), iConfig_(iConfig)
{
  //setSeperator("@@#@@"); // Change the seperator used in DB
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
bool SiPixelHistoryDQMService::setDBLabelsForUser  (std::string& keyName, std::vector<std::string>& userDBContent, std::string& quantity){
  if (quantity == "user_ymean") {
    userDBContent.push_back(keyName+fSep+std::string("yMean"));
    userDBContent.push_back(keyName+fSep+std::string("yError"));
  } else if (quantity == "user_A") {
    userDBContent.push_back(keyName+fSep+std::string("NTracksPixOverAll"));
    userDBContent.push_back(keyName+fSep+std::string("NTracksPixOverAllError"));
  } else if (quantity == "user_B") {
    userDBContent.push_back(keyName+fSep+std::string("NTracksFPixOverBPix"));
    userDBContent.push_back(keyName+fSep+std::string("NTracksFPixOverBPixError"));
  } else {
    edm::LogError("SiPixelHistoryDQMService") << "ERROR: quantity does not exist in SiPixelHistoryDQMService::setDBValuesForUser(): " << quantity;
    return false;
  }
  return true;
}
bool SiPixelHistoryDQMService::setDBValuesForUser(std::vector<MonitorElement*>::const_iterator iterMes, HDQMSummary::InputVector& values, std::string& quantity  ){

  if (quantity == "user_ymean") {
    TH1F* Hist = (TH1F*) (*iterMes)->getTH1F()->Clone();
    // if( Hist == 0 || Hist->Integral() == 0 ) {
    //   std::cout << "Error: histogram not found or empty!!" << std::endl;
    //   values.push_back( 0. );
    //   values.push_back( 0. );
    // }
    // else
    // if(  ) {
    Hist->Fit("pol0");
    TF1* Fit = Hist->GetFunction("pol0");
    float FitValue = Fit ? Fit->GetParameter(0) : 0;
    float FitError = Fit ? Fit->GetParError(0) : 0;
    std::cout << "FITERROR: " << FitError << std::endl;

    values.push_back( FitValue );
    values.push_back( FitError );
    // }
  } else if (quantity == "user_A") {
    TH1F* Hist = (TH1F*) (*iterMes)->getTH1F();
    if( Hist->GetBinContent(1) != 0 && Hist->GetBinContent(2) != 0 ) {
      values.push_back( Hist->GetBinContent(2) / Hist->GetBinContent(1) );
      values.push_back( TMath::Abs(Hist->GetBinContent(2) / Hist->GetBinContent(1)) * TMath::Sqrt( ( TMath::Power( Hist->GetBinError(1)/Hist->GetBinContent(1), 2) + TMath::Power( Hist->GetBinError(2)/Hist->GetBinContent(2), 2) )) );
    }
    else {
      values.push_back( 0. );
      values.push_back( 0. );
    }
  } else if (quantity == "user_B") {
    TH1F* Hist = (TH1F*) (*iterMes)->getTH1F();
    if( Hist->GetBinContent(3) != 0 && Hist->GetBinContent(4) != 0 ) {
      values.push_back( Hist->GetBinContent(4) / Hist->GetBinContent(3) );
      values.push_back( TMath::Abs(Hist->GetBinContent(4) / Hist->GetBinContent(3)) * TMath::Sqrt( ( TMath::Power( Hist->GetBinError(3)/Hist->GetBinContent(3), 2) + TMath::Power( Hist->GetBinError(4)/Hist->GetBinContent(4), 2) )) );
    }
    else {
      values.push_back( 0. );
      values.push_back( 0. );
    }
  } else {
    edm::LogError("SiPixelHistoryDQMService") << "ERROR: quantity does not exist in SiPixelHistoryDQMService::setDBValuesForUser(): " << quantity;
    return false;
  }

  return true;
}

