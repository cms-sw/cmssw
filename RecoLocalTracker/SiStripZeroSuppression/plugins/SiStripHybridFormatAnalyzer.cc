// -*- C++ -*-
//
// Package:    SiStripHybridFormatAnalyzer
// Class:      SiStripHybridFormatAnalyzer
// 
/**\class SiStripHybridFormatAnalyzer SiStripHybridFormatAnalyzer.cc 

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ivan Amos Cali
//         Created:  March 20 2018
//
//
 

// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DataFormats/SiStripDigi/interface/SiStripProcessedRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"

#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"

#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripPedestalsSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripCommonModeNoiseSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingFactory.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

//ROOT inclusion
#include "TROOT.h"
#include "TFile.h"
#include "TNtuple.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TList.h"
#include "TString.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "THStack.h"


//
// class decleration
//

class SiStripHybridFormatAnalyzer : public edm::EDAnalyzer {
  public:
    explicit SiStripHybridFormatAnalyzer(const edm::ParameterSet&);
    ~SiStripHybridFormatAnalyzer() override;


  private:
    void beginJob() override ;
    void analyze(const edm::Event&, const edm::EventSetup&) override;
    void endJob() override ;

    edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > srcDigis_;
    edm::EDGetTokenT<edm::DetSetVector<SiStripProcessedRawDigi> > srcAPVCM_;
    edm::Service<TFileService> fs_;
    
    TH1F* h1Digis_;
    TH1F* h1APVCM_;
    TH1F* h1BadAPVperEvent_;
    TH1F* h1BadAPVperModule_;
    TH1F* h1BadAPVperModuleOnlyBadModule_;
    TH1F* h1Pedestals_;
    
    TCanvas* Canvas_;
    
    TFileDirectory sdDigis_;
    TFileDirectory sdMisc_;
         
    uint16_t nModuletoDisplay_;
    uint16_t actualModule_;
    
    bool plotAPVCM_;
    
    //this to plot the pedestals distribution
    edm::ESHandle<SiStripPedestals> pedestalsHandle;
    std::vector<int> pedestals;
    uint32_t peds_cache_id;
	  
	  
};


SiStripHybridFormatAnalyzer::SiStripHybridFormatAnalyzer(const edm::ParameterSet& conf){
   

  srcDigis_ = consumes<edm::DetSetVector<SiStripDigi>>(conf.getParameter<edm::InputTag>("srcDigis"));
  srcAPVCM_ = consumes<edm::DetSetVector<SiStripProcessedRawDigi>>(conf.getParameter<edm::InputTag>("srcAPVCM"));
  nModuletoDisplay_ = conf.getParameter<uint32_t>( "nModuletoDisplay" );
  plotAPVCM_ = conf.getParameter<bool>( "plotAPVCM" );

  sdDigis_= fs_->mkdir("Digis");
  sdMisc_= fs_->mkdir("Miscellanea");
  
  h1APVCM_ = sdMisc_.make<TH1F>("APV CM","APV CM", 1601, -100.5, 1500.5);
  h1APVCM_->SetXTitle("APV CM [adc]");
  h1APVCM_->SetYTitle("Entries");
  h1APVCM_->SetLineWidth(2);
  h1APVCM_->SetLineStyle(2);
  
  h1BadAPVperEvent_ = sdMisc_.make<TH1F>("BadAPV/Event","BadAPV/Event", 72786, -0.5, 72785.5);
  h1BadAPVperEvent_->SetXTitle("# Bad APVs");
  h1BadAPVperEvent_->SetYTitle("Entries");
  h1BadAPVperEvent_->SetLineWidth(2);
  h1BadAPVperEvent_->SetLineStyle(2);
 	
  h1BadAPVperModule_ = sdMisc_.make<TH1F>("BadAPV/Module","BadAPV/Module", 7, -0.5, 6.5);
  h1BadAPVperModule_->SetXTitle("# Bad APVs");
  h1BadAPVperModule_->SetYTitle("Entries");
  h1BadAPVperModule_->SetLineWidth(2);
  h1BadAPVperModule_->SetLineStyle(2);
  
  h1BadAPVperModuleOnlyBadModule_ = sdMisc_.make<TH1F>("BadAPV/Module Only Bad Modules","BadAPV/Module Only Bad Modules", 7, -0.5, 6.5);
  h1BadAPVperModuleOnlyBadModule_->SetXTitle("# Bad APVs");
  h1BadAPVperModuleOnlyBadModule_->SetYTitle("Entries");
  h1BadAPVperModuleOnlyBadModule_->SetLineWidth(2);
  h1BadAPVperModuleOnlyBadModule_->SetLineStyle(2);
  
  h1Pedestals_ = sdMisc_.make<TH1F>("Pedestals","Pedestals", 2048, -1023.5, 1023.5);
  h1Pedestals_->SetXTitle("Pedestals [adc]");
  h1Pedestals_->SetYTitle("Entries");
  h1Pedestals_->SetLineWidth(2);
  h1Pedestals_->SetLineStyle(2);  
    
}


SiStripHybridFormatAnalyzer::~SiStripHybridFormatAnalyzer()
{
 
   

}

void
SiStripHybridFormatAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  using namespace edm;

  //plotting pedestals
  //------------------------------------------------------------------
  if(actualModule_ ==0){
    uint32_t p_cache_id = es.get<SiStripPedestalsRcd>().cacheIdentifier();
    if(p_cache_id != peds_cache_id) {
      es.get<SiStripPedestalsRcd>().get(pedestalsHandle);
      peds_cache_id = p_cache_id;
    }
    std::vector<uint32_t> detIdV;
    pedestalsHandle->getDetIds(detIdV);
  
    for(uint32_t i=0; i < detIdV.size(); ++i){
      pedestals.clear();
      SiStripPedestals::Range pedestalsRange = pedestalsHandle->getRange(detIdV[i]);
      pedestals.resize((pedestalsRange.second- pedestalsRange.first)*8/10);
      pedestalsHandle->allPeds(pedestals, pedestalsRange);
      for(uint32_t it=0; it < pedestals.size(); ++it) h1Pedestals_->Fill(pedestals[it]);
    }
  }
     
  //plotting CMN
  //------------------------------------------------------------------
     
  if(plotAPVCM_){
    edm::Handle<edm::DetSetVector<SiStripProcessedRawDigi> > moduleCM;
    e.getByToken(srcAPVCM_,moduleCM);
	  

    edm::DetSetVector<SiStripProcessedRawDigi>::const_iterator itCMDetSetV =moduleCM->begin();
    for (; itCMDetSetV != moduleCM->end(); ++itCMDetSetV){  
      edm::DetSet<SiStripProcessedRawDigi>::const_iterator  itCM= itCMDetSetV->begin();
      for(;itCM != itCMDetSetV->end(); ++itCM) h1APVCM_->Fill(itCM->adc());
    }
  }
        
     
  //plotting digis histograms 
  //------------------------------------------------------------------
  uint32_t NBadAPVevent=0; 

  edm::Handle<edm::DetSetVector<SiStripDigi> >moduleDigis;
  e.getByToken(srcDigis_, moduleDigis);
   
     
  edm::DetSetVector<SiStripDigi>::const_iterator itDigiDetSetV =moduleDigis->begin();
  for (; itDigiDetSetV != moduleDigis->end(); ++itDigiDetSetV){ 
    uint32_t detId = itDigiDetSetV->id; 
    edm::RunNumber_t const run = e.id().run();
    edm::EventNumber_t const event = e.id().event();
    
    char detIds[20];
    char evs[20];
    char runs[20]; 
   
    if(actualModule_ < nModuletoDisplay_){
      sprintf(detIds,"%ul", detId);
      sprintf(evs,"%llu", event);
      sprintf(runs,"%u", run);
      char* dHistoName = Form("Id_%s_run_%s_ev_%s",detIds, runs, evs);
      h1Digis_ = sdDigis_.make<TH1F>(dHistoName,dHistoName, 768, -0.5, 767.5); 
      h1Digis_->SetXTitle("strip #");
      h1Digis_->SetYTitle("adc");
      h1Digis_->SetLineWidth(2);
      h1Digis_->SetLineStyle(2);
    }
    uint16_t stripsPerAPV[6]={0,0,0,0,0,0};
    edm::DetSet<SiStripDigi>::const_iterator  itDigi= itDigiDetSetV->begin();
    for(;itDigi != itDigiDetSetV->end(); ++itDigi){
      uint16_t strip = itDigi->strip();
      uint16_t adc = itDigi->adc();
      if(actualModule_ < nModuletoDisplay_) h1Digis_->Fill(strip, adc);
      actualModule_++;
      //std::cout << "detID " << detId << " strip " << strip << " adc " << adc << std::endl;
    	
      stripsPerAPV[strip/128]++;
    }
    	
    uint16_t NBadAPVmodule=0;
    for(uint16_t APVn=0; APVn<6; APVn++){
      if(stripsPerAPV[APVn]>64){
        NBadAPVevent++;
        NBadAPVmodule++;
      }
    }
    h1BadAPVperModule_->Fill(NBadAPVmodule);
    if(NBadAPVmodule) h1BadAPVperModuleOnlyBadModule_->Fill(NBadAPVmodule);
  }
  h1BadAPVperEvent_->Fill(NBadAPVevent);

}


// ------------ method called once each job just before starting event loop  ------------
void SiStripHybridFormatAnalyzer::beginJob()
{
  
  
  actualModule_ =0;  
   
 
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiStripHybridFormatAnalyzer::endJob() {
     
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripHybridFormatAnalyzer);

