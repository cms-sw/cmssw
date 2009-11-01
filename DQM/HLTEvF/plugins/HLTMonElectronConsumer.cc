#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/HLTEvF/interface/HLTMonElectronConsumer.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

using namespace edm;

HLTMonElectronConsumer::HLTMonElectronConsumer(const edm::ParameterSet& iConfig)
{
  
  LogDebug("HLTMonElectronConsumer") << "constructor...." ;
  
  logFile_.open("HLTMonElectronConsumer.log");
  
  dbe = NULL;
  if (iConfig.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe = Service < DQMStore > ().operator->();
    dbe->setVerbose(0);
  }
  
  outputFile_ =
    iConfig.getUntrackedParameter <std::string>("outputFile", "");
  if (outputFile_.size() != 0) {
    LogInfo("HLTMonElectronConsumer") << "L1T Monitoring histograms will be saved to " 
			      << outputFile_ ;
  }
  else {
    outputFile_ = "L1TDQM.root";
  }
  
  bool disable =
    iConfig.getUntrackedParameter < bool > ("disableROOToutput", false);
  if (disable) {
    outputFile_ = "";
  }
  
  pixeltag_=iConfig.getParameter<edm::InputTag>("PixelTag");
  isotag_=iConfig.getParameter<edm::InputTag>("IsoTag");
  
  dirname_="HLT/HLTMonElectron/"+iConfig.getParameter<std::string>("@module_label");
  pixeldirname_="HLT/HLTMonElectron/"+pixeltag_.label();
  isodirname_="HLT/HLTMonElectron/"+isotag_.label();
  
  if (dbe != NULL) {
    dbe->setCurrentFolder(dirname_);
  }
  
}


HLTMonElectronConsumer::~HLTMonElectronConsumer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
HLTMonElectronConsumer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //total pixelmatch efficiencies from summary histo
  if(pixeltotal!=NULL){
    LogInfo("HLTMonElectronConsumer") << "  pixelhisto  " <<  pixeltotal->getBinContent(1);
    if(pixeltotal->getBinContent(1)!=0)
      pixelEff->Fill(pixeltotal->getBinContent(2)/pixeltotal->getBinContent(1));
    else
      pixelEff->Fill(0.);
    
    if(pixeltotal->getBinContent(3)!=0)
      trackEff->Fill(pixeltotal->getBinContent(4)/pixeltotal->getBinContent(3));
    else
      trackEff->Fill(0.);

    // efficiency as kinematic function
    for(int i =0; i<2 ;i++){
      TH1F* num;
      TH1F* denom;
      num=pixelhistosEt[2*i+1]->getTH1F();
      denom=pixelhistosEt[2*i]->getTH1F();
      for(int j=1; j <= pixelhistosEtOut[i]->getNbinsX();j++ ){
	if(denom->GetBinContent(j)!=0)
	  pixelhistosEtOut[i]->setBinContent(j,num->GetBinContent(j)/denom->GetBinContent(j));
	else
	  pixelhistosEtOut[i]->setBinContent(j,0.);
      }
      num=pixelhistosEta[2*i+1]->getTH1F();
      denom=pixelhistosEta[2*i]->getTH1F();
      for(int j=1; j <= pixelhistosEtaOut[i]->getNbinsX();j++ ){
	if(denom->GetBinContent(j)!=0)
	  pixelhistosEtaOut[i]->setBinContent(j,num->GetBinContent(j)/denom->GetBinContent(j));
	else
	  pixelhistosEtaOut[i]->setBinContent(j,0.);
      }
      num=pixelhistosPhi[2*i+1]->getTH1F();
      denom=pixelhistosPhi[2*i]->getTH1F();
      for(int j=1; j <= pixelhistosPhiOut[i]->getNbinsX();j++ ){
	if(denom->GetBinContent(j)!=0)
	  pixelhistosPhiOut[i]->setBinContent(j,num->GetBinContent(j)/denom->GetBinContent(j));
	else
	  pixelhistosPhiOut[i]->setBinContent(j,0.);
      }
    }
  }else
    LogInfo("HLTMonElectronConsumer") << " empty pixelhisto  " ;

  if(isototal!=NULL){
    TH1F* refhist = isototal->getTH1F();
    for(int i =1; i<= refhist->GetNbinsX();i++){      
      if(refhist->GetMaximum(i)!=0)
	isocheck->setBinContent(i,refhist->GetBinContent(i)/refhist->GetMaximum());
      else
	isocheck->setBinContent(i,0.);
    } 
  }else
    LogInfo("HLTMonElectronConsumer") << " empty isohisto  " ;
  


}


// ------------ method called once each job just before starting event loop  ------------
void 
HLTMonElectronConsumer::beginJob(const edm::EventSetup&)
{
  
  DQMStore *dbe = 0;
  dbe = Service < DQMStore > ().operator->();
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);
    dbe->rmdir(dirname_);
  }
  
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);
    

    // load pixel MEs
    std::string tmpname = pixeldirname_ + "/total eff";
    LogInfo("HLTMonElectronConsumer") << " reading histo: "  << tmpname;    
    pixeltotal=dbe->get(tmpname);
    TH1F* refhist;
    if(pixeltotal!=0){
      for(int i = 0; i<4; i++){
	LogInfo("HLTMonElectronConsumer") << "loop iteration: "<<i ;
	refhist=pixeltotal->getTH1F();
       	LogInfo("HLTMonElectronConsumer") << "retrieving: " <<  pixeldirname_ + "/" + refhist->GetXaxis()->GetBinLabel(i+1) + "eta";
	tmpname = pixeldirname_ + "/" + refhist->GetXaxis()->GetBinLabel(i+1) + "eta";
	pixelhistosEta[i]=dbe->get(tmpname);
	tmpname = pixeldirname_ + "/" + refhist->GetXaxis()->GetBinLabel(i+1) + "et";
	pixelhistosEt[i]=dbe->get(tmpname);
	tmpname = pixeldirname_ + "/" + refhist->GetXaxis()->GetBinLabel(i+1) + "phi";
	pixelhistosPhi[i]=dbe->get(tmpname);
      }
      LogInfo("HLTMonElectronConsumer") << "Et ";
      refhist = pixelhistosEt[0]->getTH1F();
      pixelhistosEtOut[0]  =dbe->book1D("pixel eff et","pixel eff et",refhist->GetNbinsX(),refhist->GetXaxis()->GetXmin(),refhist->GetXaxis()->GetXmax());
      pixelhistosEtOut[1]  =dbe->book1D("track eff et","track eff et",refhist->GetNbinsX(),refhist->GetXaxis()->GetXmin(),refhist->GetXaxis()->GetXmax());
      LogInfo("HLTMonElectronConsumer") << "Eta ";
      refhist = pixelhistosEta[0]->getTH1F();
      pixelhistosEtaOut[0] =dbe->book1D("pixel eff eta","pixel eff eta",refhist->GetNbinsX(),refhist->GetXaxis()->GetXmin(),refhist->GetXaxis()->GetXmax());
      pixelhistosEtaOut[1] =dbe->book1D("track eff eta","track eff eta",refhist->GetNbinsX(),refhist->GetXaxis()->GetXmin(),refhist->GetXaxis()->GetXmax());
      LogInfo("HLTMonElectronConsumer") << "Phi ";
      refhist = pixelhistosPhi[0]->getTH1F();
      pixelhistosPhiOut[0] =dbe->book1D("pixel eff phi","pixel eff phi",refhist->GetNbinsX(),refhist->GetXaxis()->GetXmin(),refhist->GetXaxis()->GetXmax());
      pixelhistosPhiOut[1] =dbe->book1D("track eff phi","track eff phi",refhist->GetNbinsX(),refhist->GetXaxis()->GetXmin(),refhist->GetXaxis()->GetXmax());
    }else
      LogInfo("HLTMonElectronConsumer") << "pixelhisto doesn't exist during beginJob" ;
      
    pixelEff=dbe->bookFloat("total pixelmatch");
    trackEff=dbe->bookFloat("total trackmatch");

    LogInfo("HLTMonElectronConsumer") << "writing: "  << pixelEff->getPathname();


    tmpname = isodirname_ + "/total eff";
    LogInfo("HLTMonElectronConsumer") << " reading histo: "  << tmpname;    
    isototal=dbe->get(tmpname);
    refhist = isototal->getTH1F();
    isocheck = dbe->book1D("consistency check","consistency check",refhist->GetNbinsX(),refhist->GetXaxis()->GetXmin(),refhist->GetXaxis()->GetXmax());
    

  } // end "if(dbe)"
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HLTMonElectronConsumer::endJob() {

//     std::cout << "HLTMonElectronConsumer: end job...." << std::endl;
 
   if (outputFile_.size() != 0 && dbe)
     dbe->save(outputFile_);
 
   return;
}

//DEFINE_FWK_MODULE(HLTMonElectronConsumer);
