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
      ~SiStripHybridFormatAnalyzer();


   private:
      virtual void beginJob() override ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override ;
      
          edm::InputTag srcDigis_;
        
  edm::Service<TFileService> fs_;
	  
	    TH1F* h1Digis_;
	  TCanvas* Canvas_;
	  std::vector<TH1F> vProcessedRawDigiHisto_;
	  
	  
	  uint16_t nModuletoDisplay_;
	  uint16_t actualModule_;
};


SiStripHybridFormatAnalyzer::SiStripHybridFormatAnalyzer(const edm::ParameterSet& conf){
   

  srcDigis_ =  conf.getParameter<edm::InputTag>( "srcDigis" );
  nModuletoDisplay_ = conf.getParameter<uint32_t>( "nModuletoDisplay" );


 
}


SiStripHybridFormatAnalyzer::~SiStripHybridFormatAnalyzer()
{
 
   

}

void
SiStripHybridFormatAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& es)
{
   using namespace edm;

     TFileDirectory sdDigis_= fs_->mkdir("Digis");

     edm::Handle<edm::DetSetVector<SiStripDigi> >moduleDigis;
   	 e.getByLabel(srcDigis_, moduleDigis); 
   
   
     edm::DetSetVector<SiStripDigi>::const_iterator itDigiDetSetV =moduleDigis->begin();
     for (; itDigiDetSetV != moduleDigis->end(); ++itDigiDetSetV){ 
		uint32_t detId = itDigiDetSetV->id; 
      	edm::RunNumber_t const run = e.id().run();
      	edm::EventNumber_t const event = e.id().event();
       
      	char detIds[20];
   	  	char evs[20];
      	char runs[20]; 
      
      	sprintf(detIds,"%ul", detId);
      	sprintf(evs,"%llu", event);
      	sprintf(runs,"%u", run);
      	char* dHistoName = Form("Id_%s_run_%s_ev_%s",detIds, runs, evs);
      	h1Digis_ = sdDigis_.make<TH1F>(dHistoName,dHistoName, 768, -0.5, 767.5); 
      
       
       	edm::DetSet<SiStripDigi>::const_iterator  itDigi= itDigiDetSetV->begin();
      	for(;itDigi != itDigiDetSetV->end(); ++itDigi){
      	    uint16_t strip = itDigi->strip();
      	    uint16_t adc = itDigi->adc();
       		h1Digis_->Fill(strip, adc);
       		
       		//std::cout << "detID " << detId << " strip " << strip << " adc " << adc << std::endl;
       	}
    }
   
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

