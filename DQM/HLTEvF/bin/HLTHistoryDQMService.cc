#include "DQM/HLTEvF/bin/HLTHistoryDQMService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Diagnostic/interface/HDQMfitUtilities.h"


HLTHistoryDQMService::HLTHistoryDQMService(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg)
: DQMHistoryServiceBase::DQMHistoryServiceBase(iConfig, aReg), iConfig_(iConfig)
{
  edm::LogInfo("HLTHistoryDQMService") <<  "[HLTHistoryDQMService::HLTHistoryDQMService]";
  threshold_ = iConfig.getUntrackedParameter<double>("threshold", 0.);

}


HLTHistoryDQMService::~HLTHistoryDQMService() { 
  edm::LogInfo("HLTHistoryDQMService") <<  "[HLTHistoryDQMService::~HLTHistoryDQMService]";
}


uint32_t HLTHistoryDQMService::returnDetComponent(const MonitorElement* ME){
  LogTrace("HLTHistoryDQMService") <<  "[HLTHistoryDQMService::returnDetComponent]";
//  std::string str=ME->getName();
//  size_t __key_length__=7;
//  size_t __detid_length__=9;


//  if(str.find("__det__")!= std::string::npos){
//    return atoi(str.substr(str.find("__det__")+__key_length__,__detid_length__).c_str());
//  }
    return 2;

}

//distinguere histo in base al path ed assegnare numero e poi plottare sulla base del numero.

//Example on how to define an user function for the statistic extraction
bool HLTHistoryDQMService::setDBLabelsForUser  (std::string& keyName, std::vector<std::string>& userDBContent){
  userDBContent.push_back(keyName+std::string("@")+std::string("plateau"));
  userDBContent.push_back(keyName+std::string("@")+std::string("eplateau"));
  userDBContent.push_back(keyName+std::string("@")+std::string("usrMean"));
  userDBContent.push_back(keyName+std::string("@")+std::string("usrRMS"));
  return true;
}
bool HLTHistoryDQMService::setDBValuesForUser(std::vector<MonitorElement*>::const_iterator iterMes, HDQMSummary::InputVector& values  ){

// efficiency at pleteau
  Double_t plateau=0.;
  Double_t eplateau=0.;
  Double_t wdenom=0.;
  Double_t sig_;

  for(int ibin=1; ibin<= (*iterMes)->getTH1F()->GetXaxis()->GetNbins();ibin++) {
    if((*iterMes)->getTH1F()->GetXaxis()->GetBinCenter(ibin)> threshold_) { 
       if((*iterMes)->getTH1F()->GetBinContent(ibin)> 0.) { 
         sig_ = (*iterMes)->getTH1F()->GetBinError(ibin);
         plateau+=(*iterMes)->getTH1F()->GetBinContent(ibin)/(sig_*sig_); 
	 wdenom += 1./(sig_*sig_);
      }
    }
  }
  
//  plateau=plateau/wdenom;
  if(wdenom > 0.) {
     plateau=plateau/wdenom;
     eplateau=sqrt(1/wdenom);
   } else {
     plateau=0.;
     eplateau=0.;
  }
       
  values.push_back(plateau);
  values.push_back(eplateau);

// usrMean

  Double_t eventsCount=0.;
  Double_t usrMean=0.;
  for(int ibin=1; ibin<= (*iterMes)->getTH1F()->GetXaxis()->GetNbins();ibin++) {
         usrMean+=(*iterMes)->getTH1F()->GetBinContent(ibin)*(*iterMes)->getTH1F()->GetXaxis()->GetBinCenter(ibin);
	 eventsCount+=(*iterMes)->getTH1F()->GetBinContent(ibin);
  }
  usrMean= usrMean/eventsCount;
  values.push_back( usrMean );

  eventsCount=0.;
  Double_t usrRMS=0.;
  Double_t di=0.;
  for(int ibin=1; ibin<= (*iterMes)->getTH1F()->GetXaxis()->GetNbins();ibin++) {
       if((*iterMes)->getTH1F()->GetBinContent(ibin)> 0.) { 
         di = (*iterMes)->getTH1F()->GetXaxis()->GetBinCenter(ibin);
         usrRMS+=(di-usrMean)*(di-usrMean);
	 eventsCount+=(*iterMes)->getTH1F()->GetBinContent(ibin);
       }
  }
  usrRMS= sqrt(usrRMS/eventsCount);
  values.push_back( usrRMS );
  return true;
}

