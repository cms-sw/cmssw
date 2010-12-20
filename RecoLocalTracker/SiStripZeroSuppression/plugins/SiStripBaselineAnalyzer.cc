// -*- C++ -*-
//
// Package:    SiStripBaselineAnalyzer
// Class:      SiStripBaselineAnalyzer
// 
/**\class SiStripBaselineAnalyzer SiStripBaselineAnalyzer.cc Validation/SiStripAnalyzer/src/SiStripBaselineAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ivan Amos Cali
//         Created:  Mon Jul 28 14:10:52 CEST 2008
// $Id$
//
//
 

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripProcessedRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripPedestalsSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripCommonModeNoiseSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingFactory.h"

//ROOT inclusion
#include "TROOT.h"
#include "TFile.h"
#include "TNtuple.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TH2F.h"


//
// class decleration
//

class SiStripBaselineAnalyzer : public edm::EDAnalyzer {
   public:
      explicit SiStripBaselineAnalyzer(const edm::ParameterSet&);
      ~SiStripBaselineAnalyzer();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
	  std::auto_ptr<SiStripPedestalsSubtractor>   subtractorPed_;
   
	  std::string outputFile_;
	  edm::InputTag srcBaseline_;
	  edm::InputTag srcProcessedRawDigi_;
      
	  TH1F* h1ProcessedRawDigis;
	  TH1F* h1Baseline;
	  
	  TFile* oFile_;
	  std::vector<TH1F> vProcessedRawDigiHisto;
	  std::vector<TH1F> vBaselineHisto;
	  
	  uint16_t nModuletoDisplay;
	  uint16_t actualModule;
};


SiStripBaselineAnalyzer::SiStripBaselineAnalyzer(const edm::ParameterSet& conf){
   
  outputFile_ = conf.getUntrackedParameter<std::string>("outputFile", "StripHistos.root");
  srcBaseline_ =  conf.getParameter<edm::InputTag>( "srcBaseline" );
  srcProcessedRawDigi_ =  conf.getParameter<edm::InputTag>( "srcProcessedRawDigi" );
  subtractorPed_ = SiStripRawProcessingFactory::create_SubtractorPed(conf.getParameter<edm::ParameterSet>("Algorithms"));
  nModuletoDisplay = conf.getParameter<uint32_t>( "nModuletoDisplay" );
  
  vProcessedRawDigiHisto.clear();
  vProcessedRawDigiHisto.reserve(10000);
  
  vBaselineHisto.clear();
  vBaselineHisto.reserve(10000);
  //nModuletoDisplay =100;
}


SiStripBaselineAnalyzer::~SiStripBaselineAnalyzer()
{
 
   

}

void
SiStripBaselineAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& es)
{
   using namespace edm;
   
   
   subtractorPed_->init(es);
   
   edm::Handle<edm::DetSetVector<SiStripProcessedRawDigi> > moduleBaseline;
   e.getByLabel(srcBaseline_, moduleBaseline);
   
   edm::Handle< edm::DetSetVector<SiStripRawDigi> > moduleRawDigi;
   e.getByLabel(srcProcessedRawDigi_,moduleRawDigi);

   
   //vDigiHisto.clear();
   //vDigiHisto.reserve(10000);
  
   //vBaselineHisto.clear();
   //vBaselineHisto.reserve(10000);
  
   char detIds[20];
   char evs[20];
   char runs[20];    
    
   edm::DetSetVector<SiStripProcessedRawDigi>::const_iterator itDSBaseline = moduleBaseline->begin();
   edm::DetSetVector<SiStripRawDigi>::const_iterator itRawDigis = moduleRawDigi->begin();
    
   std::cout<< "Number of module with HIP in this event: " << moduleRawDigi->size() << std::endl;
   for (; itRawDigis != moduleRawDigi->end(); ++itRawDigis, ++itDSBaseline) {
      if(actualModule > nModuletoDisplay) return;
      uint32_t detId = itRawDigis->id;
	  
	  
      if(itDSBaseline->id != detId){
		std::cout << "Collections out of Synch. Something of fishy is going on ;-)" << std::endl;
		return;
      }	  
      
	  actualModule++;
	  uint32_t event = e.id().event();
	  uint32_t run = e.id().run();
	  std::cout << "processing module N: " << actualModule<< " detId: " << detId << " event: "<< event << std::endl; 
	  
	  sprintf(detIds,"%ul", detId);
	  sprintf(evs,"%ul", event);
	  sprintf(runs,"%ul", run);
	  char* dHistoName = Form("Id:%s_run:%s_ev:%s",detIds, runs, evs);
	  //char* bHistoName = Form("Id:%s_run:%s_ev:%s",detIds, runs, evs);
      h1ProcessedRawDigis = new TH1F(dHistoName,dHistoName, 768, -0.5, 767.5); 
	  h1Baseline = new TH1F(dHistoName,dHistoName, 768, -0.5, 767.5); 
      
	  
	  h1ProcessedRawDigis->SetXTitle("strip#");  
	  h1ProcessedRawDigis->SetYTitle("ADC");
	  h1ProcessedRawDigis->SetMaximum(1024.);
      h1ProcessedRawDigis->SetMinimum(-300.);
	  h1ProcessedRawDigis->SetLineWidth(2);

   
     h1Baseline->SetXTitle("strip#");
     h1Baseline->SetYTitle("ADC");
     h1Baseline->SetMaximum(1024.);
     h1Baseline->SetMinimum(-300.);
     h1Baseline->SetLineWidth(2);
	 h1Baseline->SetLineStyle(2);
	 h1Baseline->SetLineColor(2);
	 
	  
	  edm::DetSet<SiStripProcessedRawDigi>::const_iterator  itBaseline; 
	  std::vector<int16_t>::const_iterator itProcessedRawDigis;
	  
	  
	  std::vector<int16_t> ProcessedRawDigis(itRawDigis->size());
	  //Digis.clear();
	  //for(itDigis = itRawDigis->begin();itDigis != itRawDigis->end(); ++itDigis) Digis.push_back((int16_t)itDigis->adc());
	  //subtractorPed_->subtract( detId,0, Digis);
	  subtractorPed_->subtract( *itRawDigis, ProcessedRawDigis);
	  
	  
	  int strip =0;
      for(itProcessedRawDigis = ProcessedRawDigis.begin(), itBaseline = itDSBaseline->begin();itProcessedRawDigis != ProcessedRawDigis.end(); ++itProcessedRawDigis, ++itBaseline){
		h1ProcessedRawDigis->Fill(strip, *itProcessedRawDigis);
		h1Baseline->Fill(strip, itBaseline->adc()); 
		++strip;
      }	  
	 
	 
	 vProcessedRawDigiHisto.push_back(*h1ProcessedRawDigis);
	 vBaselineHisto.push_back(*h1Baseline);
	 
	}
	
	
	
	 //char* CName = Form("id:%s_run:%s_ev:%s",detIds, runs, evs);
	 //char* CNameF = Form("id:%s_run:%s_ev:%s.png",detIds, runs, evs);
     //TCanvas *c1 = new TCanvas("c1",CName,700,700);
     //h1Digis->Draw("");
	 //h1Baseline->Draw("");
     //c1->SaveAs(CNameF); 
	 
    
}


// ------------ method called once each job just before starting event loop  ------------
void SiStripBaselineAnalyzer::beginJob()
{
  
  
  actualModule =0;
   
 
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiStripBaselineAnalyzer::endJob() {
    oFile_ = new TFile((const char*)outputFile_.c_str(), "RECREATE");
    oFile_->mkdir("ProcessedRawDigis");
    oFile_->mkdir("Baseline");
	
	
    std::vector<TH1F>::iterator itvProcessedRawDigis, itvBaseline; 
    itvProcessedRawDigis = vProcessedRawDigiHisto.begin();
    itvBaseline = vBaselineHisto.begin();
    
	for(; itvProcessedRawDigis != vProcessedRawDigiHisto.end(); ++itvProcessedRawDigis, ++itvBaseline){
	    //	itvBaseline->SetDirectory(oFile_->GetDirectory("Baseline"));	
		oFile_->cd("ProcessedRawDigis");
		itvProcessedRawDigis->Write();
		oFile_->cd("Baseline");
	    itvBaseline->Write();
		
	}
	oFile_->Write();
    oFile_->Close();
  
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripBaselineAnalyzer);

