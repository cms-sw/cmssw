
#include "DQMOffline/Trigger/plugins/JetPomptDQMPostProcessor.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include <iostream>
#include <string.h>
#include <iomanip>
#include<fstream>
#include <math.h>


JetPomptDQMPostProcessor::JetPomptDQMPostProcessor(const edm::ParameterSet& pset)
{
  subDir_ = pset.getUntrackedParameter<std::string>("subDir");
}

void JetPomptDQMPostProcessor::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter)
{
  //////////////////////////////////
  // setup DQM stor               //
  //////////////////////////////////

  bool isPFJetDir = false;
  bool isCaloJetDir = false;
  //go to the directory to be processed
  if(igetter.dirExists(subDir_)) ibooker.cd(subDir_);
  else {
    edm::LogWarning("JetPomptDQMPostProcessor") << "cannot find directory: " << subDir_ << " , skipping";
    return;
  }
  
  std::vector<std::string> subdirectories = igetter.getSubdirs();

  for(std::vector<std::string>::iterator dir = subdirectories.begin() ;dir!= subdirectories.end(); dir++ ){

    ibooker.cd(*dir);
    isPFJetDir = false;
    isCaloJetDir = false;
    if (TString(*dir).Contains("HLT_PFJet")) isPFJetDir = true;
    if (TString(*dir).Contains("HLT_CaloJet")) isCaloJetDir = true;
    if (isPFJetDir) {
      dividehistos(ibooker, igetter,  "effic_pfjetabseta_HEP17",      "effic_pfjetabseta_HEM17",      "ratio_pfjeteta_HEP17VSHEM17",       "| #eta |; Ratio","HEP17/HEM17 vs |#eta|","1D");// numer deno 
      dividehistos(ibooker, igetter,  "effic_pfjetpT_HEP17",          "effic_pfjetpT_HEM17",          "ratio_pfjetpT_HEP17VSHEM17",        " pT [GeV]; Ratio","HEP17/HEM17 vs pT","1D");// numer deno 
      dividehistos(ibooker, igetter,  "effic_pfjetpT_HEP17_pTThresh", "effic_pfjetpT_HEM17_pTThresh", "ratio_pfjetpT_pTTresh_HEP17VSHEM17"," pT [GeV]; Ratio","HEP17/HEM17 vs pT","1D");// numer deno 
      dividehistos(ibooker, igetter,  "effic_pfjetphi_HEP17",         "effic_pfjetphi_HEM17",         "ratio_pfjetphi_HEP17VSHEM17",       " #phi; Ratio","HEP17/HEM17 vs #phi","1D");// numer deno 
    }
    if (isCaloJetDir) {
      dividehistos(ibooker, igetter,  "effic_calojetabseta_HEP17",      "effic_calojetabseta_HEM17",      "ratio_calojeteta_HEP17VSHEM17","| #eta |; Ratio","HEP17/HEM17 vs |#eta|","1D");// numer deno 
      dividehistos(ibooker, igetter,  "effic_calojetpT_HEP17",          "effic_calojetpT_HEM17",          "ratio_calojetpT_HEP17VSHEM17"," pT [GeV]; Ratio","HEP17/HEM17 vs pT","1D");// numer deno 
      dividehistos(ibooker, igetter,  "effic_calojetpT_HEP17_pTThresh", "effic_calojetpT_HEM17_pTThresh", "ratio_calojetpT_pTTresh_HEP17VSHEM17"," pT [GeV]; Ratio","HEP17/HEM17 vs pT","1D");// numer deno 
      dividehistos(ibooker, igetter,  "effic_calojetphi_HEP17",         "effic_calojetphi_HEM17",         "ratio_calojetphi_HEP17VSHEM17"," #phi; Ratio","HEP17/HEM17 vs #phi","1D");// numer deno 
    }
   
    ibooker.goUp();         
  }
}

//----------------------------------------------------------------------
//TProfile* 
//TH1F* 
void
JetPomptDQMPostProcessor::dividehistos(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, const std::string& numName, const std::string& denomName, 
				     const std::string& outName, const std::string& label, const std::string& titel,const std::string histDim)
{
  //ibooker.pwd();
  if (histDim == "1D"){
    TH1F* num = getHistogram(ibooker, igetter, ibooker.pwd()+"/"+numName);
    
    TH1F* denom = getHistogram(ibooker, igetter, ibooker.pwd()+"/"+denomName);
    
    if (num == NULL)
      edm::LogWarning("JetPomptDQMPostProcessor") << "numerator histogram " << ibooker.pwd()+"/"+numName << " does not exist";
    if (denom == NULL)
      edm::LogWarning("JetPomptDQMPostProcessor") << "denominator histogram " << ibooker.pwd()+"/"+denomName << " does not exist";
    
    // Check if histograms actually exist
    //if(!num || !denom) return 0;
    if(!num || !denom) return ;
     
    MonitorElement* meOut = ibooker.book1D(outName, titel, num->GetXaxis()->GetNbins(), num->GetXaxis()->GetXmin(), num->GetXaxis()->GetXmax());
    TH1F *ratio = new TH1F("ratio", "raito", num->GetXaxis()->GetNbins(), num->GetXaxis()->GetXmin(), num->GetXaxis()->GetXmax());
    ratio->Sumw2(); 
    //ratio->Divide(num,denom,1,1,"B");
    ratio->Divide(num,denom);
    for (int i =0; i< ratio->GetNbinsX(); ++i){
       meOut->setBinContent(i+1,ratio->GetBinContent(i+1));
       meOut->setBinError(i+1,ratio->GetBinError(i+1));
    }
  }
  else if (histDim == "2D"){
    TH2F* num = getHistogram2D(ibooker, igetter, ibooker.pwd()+"/"+numName);
    
    TH2F* denom = getHistogram2D(ibooker, igetter, ibooker.pwd()+"/"+denomName);
    
    if (num == NULL)
      edm::LogWarning("JetPomptDQMPostProcessor") << "numerator histogram " << ibooker.pwd()+"/"+numName << " does not exist";
    if (denom == NULL)
      edm::LogWarning("JetPomptDQMPostProcessor") << "denominator histogram " << ibooker.pwd()+"/"+denomName << " does not exist";
    
    // Check if histograms actually exist
    //if(!num || !denom) return 0;
    if(!num || !denom) return;
    TH2F *ratio = new TH2F("ratio", "raito", num->GetXaxis()->GetNbins(), num->GetXaxis()->GetXmin(), num->GetXaxis()->GetXmax(),num->GetYaxis()->GetNbins(), num->GetYaxis()->GetXmin(), num->GetYaxis()->GetXmax());
    ratio->Sumw2(); 
    MonitorElement* meOut = ibooker.book2D(outName, titel, num->GetXaxis()->GetNbins(), num->GetXaxis()->GetXmin(), num->GetXaxis()->GetXmax(), num->GetYaxis()->GetNbins(), num->GetYaxis()->GetXmin(), num->GetYaxis()->GetXmax());
        
//    num->Divide(num,denom,1,1,"B");
    ratio->Divide(num,denom);
    for (int i =0; i< ratio->GetNbinsX(); ++i){
       for (int j = 0; j < ratio->GetNbinsY();++i){
          meOut->setBinContent(i+1,j+1,ratio->GetBinContent(i+1,j+1));
          meOut->setBinError(i+1,j+1,ratio->GetBinError(i+1,j+1));
       }
    }
  }
  else {
      edm::LogWarning("JetPomptDQMPostProcessor") << "CHECK OUT RATIO DIMESION OF RATIO !!";
  }
}

//----------------------------------------------------------------------
TH1F *
JetPomptDQMPostProcessor::getHistogram(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, const std::string &histoPath)
{
  ibooker.pwd();
  MonitorElement *monElement = igetter.get(histoPath);
  if (monElement != NULL)
    return monElement->getTH1F();
  else
    return NULL;
}
TH2F *
JetPomptDQMPostProcessor::getHistogram2D(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, const std::string &histoPath)
{
  ibooker.pwd();
  MonitorElement *monElement = igetter.get(histoPath);
  if (monElement != NULL)
    return monElement->getTH2F();
  else
    return NULL;
}
//----------------------------------------------------------------------

DEFINE_FWK_MODULE(JetPomptDQMPostProcessor);
