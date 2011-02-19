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
// $Id: SiStripBaselineAnalyzer.cc,v 1.2 2010/11/14 22:54:37 edwenger Exp $
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

#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripPedestalsSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripCommonModeNoiseSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingFactory.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

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

class SiStripBaselineAnalyzer : public edm::EDAnalyzer {
   public:
      explicit SiStripBaselineAnalyzer(const edm::ParameterSet&);
      ~SiStripBaselineAnalyzer();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
	  std::auto_ptr<SiStripPedestalsSubtractor>   subtractorPed_;
      bool plotClusters_;
      bool plotBaseline_;
      bool plotRawDigi_;
	  
	  edm::InputTag srcBaseline_;
	  edm::InputTag srcProcessedRawDigi_;
      
      edm::Service<TFileService> fs_;
	  TH1F* h1BadAPVperEvent_;
	  
	  TH1F* h1ProcessedRawDigis_;
	  TH1F* h1Baseline_;
	  TH1F* h1Clusters_;
	  	  
	  
	  TCanvas* Canvas_;
	  std::vector<TH1F> vProcessedRawDigiHisto_;
	  std::vector<TH1F> vBaselineHisto_;
	  std::vector<TH1F> vClusterHisto_;
	  
	  uint16_t nModuletoDisplay_;
	  uint16_t actualModule_;
};


SiStripBaselineAnalyzer::SiStripBaselineAnalyzer(const edm::ParameterSet& conf){
   
  srcBaseline_ =  conf.getParameter<edm::InputTag>( "srcBaseline" );
  srcProcessedRawDigi_ =  conf.getParameter<edm::InputTag>( "srcProcessedRawDigi" );
  subtractorPed_ = SiStripRawProcessingFactory::create_SubtractorPed(conf.getParameter<edm::ParameterSet>("Algorithms"));
  nModuletoDisplay_ = conf.getParameter<uint32_t>( "nModuletoDisplay" );
  plotClusters_ = conf.getParameter<bool>( "plotClusters" );
  plotBaseline_ = conf.getParameter<bool>( "plotBaseline" );
  plotRawDigi_ = conf.getParameter<bool>( "plotRawDigi" );
  
  
  h1BadAPVperEvent_ = fs_->make<TH1F>("BadAPV/Event","BadAPV/Event", 2001, -0.5, 2000.5);
  h1BadAPVperEvent_->SetXTitle("# Modules with Bad APVs");
  h1BadAPVperEvent_->SetYTitle("Entries");
  h1BadAPVperEvent_->SetLineWidth(2);
  h1BadAPVperEvent_->SetLineStyle(2);
 
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
  
   
   	edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusters;
   if(plotClusters_){
   	edm::InputTag clusLabel("siStripClusters");
   	e.getByLabel(clusLabel, clusters);
   }
   
   char detIds[20];
   char evs[20];
   char runs[20];    
   
   TFileDirectory sdProcessedRawDigis_= fs_->mkdir("ProcessedRawDigis");
   TFileDirectory sdBaseline_= fs_->mkdir("Baseline");
   TFileDirectory sdClusters_= fs_->mkdir("Clusters");
 
   edm::DetSetVector<SiStripProcessedRawDigi>::const_iterator itDSBaseline = moduleBaseline->begin();
   edm::DetSetVector<SiStripRawDigi>::const_iterator itRawDigis = moduleRawDigi->begin();
   
   uint32_t NBabAPVs = moduleRawDigi->size();     
   std::cout<< "Number of module with HIP in this event: " << NBabAPVs << std::endl;
   h1BadAPVperEvent_->Fill(NBabAPVs);
   
   for (; itRawDigis != moduleRawDigi->end(); ++itRawDigis, ++itDSBaseline) {
      if(actualModule_ > nModuletoDisplay_) return;
      uint32_t detId = itRawDigis->id;
	  
	  //std::cout << "bas id: " << itDSBaseline->id << " raw id: " << detId << std::endl;
      if(itDSBaseline->id != detId){
		std::cout << "Collections out of Synch. Something of fishy is going on ;-)" << std::endl;
		return;
      }	  
      
	  actualModule_++;
	  uint32_t event = e.id().event();
	  uint32_t run = e.id().run();
	  //std::cout << "processing module N: " << actualModule_<< " detId: " << detId << " event: "<< event << std::endl; 
	  

	 
	
	  
	  edm::DetSet<SiStripProcessedRawDigi>::const_iterator  itBaseline; 
	  std::vector<int16_t>::const_iterator itProcessedRawDigis;
	  
	  edm::DetSet<SiStripRawDigi>::const_iterator itRaw = itRawDigis->begin(); 
	  bool restAPV[6] = {0,0,0,0,0,0};
	  int strip =0, totADC=0;
      int minAPVRes = 7, maxAPVRes = -1;
      for(;itRaw != itRawDigis->end(); ++itRaw, ++strip){
      	float adc = itRaw->adc();
      	totADC+= adc;
      	if(strip%127 ==0){
      		//std::cout << "totADC " << totADC << std::endl;
      	    int APV = strip/128;
      	    if(totADC!= 0){
      	    	restAPV[APV] = true;
      			totADC =0;
      			if(APV>maxAPVRes) maxAPVRes = APV;
      			if(APV<minAPVRes) minAPVRes = APV;
      		}
      	}
      }

      uint16_t bins =768;
      float minx = -0.5, maxx=767.5;
      if(minAPVRes !=7){
      	minx = minAPVRes * 128 -0.5;
      	maxx = maxAPVRes * 128 + 127.5;
      	bins = maxx-minx;
      }
	  sprintf(detIds,"%ul", detId);
	  sprintf(evs,"%ul", event);
	  sprintf(runs,"%ul", run);
	  char* dHistoName = Form("Id:%s_run:%s_ev:%s",detIds, runs, evs);
	  h1ProcessedRawDigis_ = sdProcessedRawDigis_.make<TH1F>(dHistoName,dHistoName, bins, minx, maxx); 
	  h1Baseline_ = sdBaseline_.make<TH1F>(dHistoName,dHistoName, bins, minx, maxx); 
      
      if(plotClusters_){
        h1Clusters_ = sdClusters_.make<TH1F>(dHistoName,dHistoName, bins, minx, maxx);
	  
        h1Clusters_->SetXTitle("strip#");
        h1Clusters_->SetYTitle("ADC");
        h1Clusters_->SetMaximum(1024.);
        h1Clusters_->SetMinimum(-300.);
        h1Clusters_->SetLineWidth(2);
	    h1Clusters_->SetLineStyle(2);
	    h1Clusters_->SetLineColor(3);
	  }

	  h1ProcessedRawDigis_->SetXTitle("strip#");  
	  h1ProcessedRawDigis_->SetYTitle("ADC");
	  h1ProcessedRawDigis_->SetMaximum(1024.);
      h1ProcessedRawDigis_->SetMinimum(-300.);
	  h1ProcessedRawDigis_->SetLineWidth(2);

   
      h1Baseline_->SetXTitle("strip#");
      h1Baseline_->SetYTitle("ADC");
      h1Baseline_->SetMaximum(1024.);
      h1Baseline_->SetMinimum(-300.);
      h1Baseline_->SetLineWidth(2);
	  h1Baseline_->SetLineStyle(2);
	  h1Baseline_->SetLineColor(2);
	  std::vector<int16_t> ProcessedRawDigis(itRawDigis->size());
	  subtractorPed_->subtract( *itRawDigis, ProcessedRawDigis);
	  
	  
	  strip =0;      
      for(itProcessedRawDigis = ProcessedRawDigis.begin(), itBaseline = itDSBaseline->begin();itProcessedRawDigis != ProcessedRawDigis.end(); ++itProcessedRawDigis, ++itBaseline){
        //for(itBaseline = itDSBaseline->begin();itBaseline != itDSBaseline->end(); ++itBaseline, ++itRaw){
		
		if(restAPV[strip/128]){
			
			// float adc = itRaw->adc();
			float adc = *itProcessedRawDigis;       
			//std::cout << "APV RESTORED!!!! " << adc << std::endl;
            h1ProcessedRawDigis_->Fill(strip, adc);
			h1Baseline_->Fill(strip, itBaseline->adc()); 
		}
		++strip;
      }	  
	  
     if(plotClusters_){
	  edmNew::DetSetVector<SiStripCluster>::const_iterator itClusters = clusters->begin();
	  for ( ; itClusters != clusters->end(); ++itClusters ){
		for ( edmNew::DetSet<SiStripCluster>::const_iterator clus =	itClusters->begin(); clus != itClusters->end(); ++clus){
            if(clus->geographicalId() == detId){
				int firststrip = clus->firstStrip();
	            //std::cout << "Found cluster in detId " << detId << " " << firststrip << " " << clus->amplitudes().size() << " -----------------------------------------------" << std::endl;		
     			strip=0;
				for( std::vector<uint8_t>::const_iterator itAmpl = clus->amplitudes().begin(); itAmpl != clus->amplitudes().end(); ++itAmpl){
                   h1Clusters_->Fill(firststrip+strip, *itAmpl);
				   ++strip;
				}
			}              
    	}
	  }
	 }
	 	 
	 
	}
		
    
}


// ------------ method called once each job just before starting event loop  ------------
void SiStripBaselineAnalyzer::beginJob()
{
  
  
  actualModule_ =0;
   
 
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiStripBaselineAnalyzer::endJob() {
     
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripBaselineAnalyzer);

