// Original Author:  Ivan Amos Cali
//         Created:  Mon Jul 28 14:10:52 CEST 2008
// $Id: SiStripBaselineValidator.cc,v 1.3 2011/11/02 00:39:33 gowdy Exp $
//
//
 

// system include files
#include <memory>
#include <iostream>

// user include files
#include "DQM/SiStripMonitorDigi/interface/SiStripBaselineValidator.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DQMServices/Core/interface/DQMStore.h"

/*#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h" 
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h" 
#include "DataFormats/TrackReco/interface/Track.h" 
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
*/

//ROOT inclusion
#include "TH1F.h"
#include "TH2F.h"
#include "TString.h"
#include "assert.h"
#include <fstream>

class TFile;



using namespace edm;
using namespace std;

SiStripBaselineValidator::SiStripBaselineValidator(const edm::ParameterSet& conf){

  srcProcessedRawDigi_ =  conf.getParameter<edm::InputTag>( "srcProcessedRawDigi" );
 // hiSelectedTracks =  conf.getParameter<edm::InputTag>( "hiSelectedTracks" );
  createOutputFile_ = conf.getUntrackedParameter<bool>("saveFile",false);
  outputFile_   = conf.getParameter<std::string>("outputFile");
  dbe = &*edm::Service<DQMStore>();




}

SiStripBaselineValidator::~SiStripBaselineValidator()
{
}

// ------------ method called once each job just before starting event loop  ------------
void SiStripBaselineValidator::beginJob()
{

 if(dbe){
    ///Setting the DQM top directories
    dbe->setCurrentFolder("SiStrip/BaselineValidator");


    h1NumbadAPVsRes_ = dbe->book1D("ResAPVs",";#ResAPVs", 100, 1.0, 10001);
    dbe->tag(h1NumbadAPVsRes_->getFullname(),1);

    h1ADC_vs_strip_ = dbe->book2D("ADCvsAPVs",";ADCvsAPVs", 768,-0.5,767.5,  1023, -0.5, 1022.5);
    dbe->tag(h1ADC_vs_strip_->getFullname(),2);


 }  
  
   return;

}

void SiStripBaselineValidator::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  /* edm::Handle<reco::TrackCollection> tracks;
   e.getByLabel("hiSelectedTracks", tracks);

   int ntracks =0;
   for (reco::TrackCollection::const_iterator track = tracks->begin(); track != tracks->end(); track++) {
   ntracks++;
    
   }
*/


  edm::Handle< edm::DetSetVector<SiStripRawDigi> > moduleRawDigi;
  e.getByLabel(srcProcessedRawDigi_,moduleRawDigi);
  edm::DetSetVector<SiStripRawDigi>::const_iterator itRawDigis = moduleRawDigi->begin();
 
  //  uint32_t Nmodule = moduleRawDigi->size();     

   int NumResAPVs=0;
   for (; itRawDigis != moduleRawDigi->end(); ++itRawDigis) {   ///loop over modules
     

     edm::DetSet<SiStripRawDigi>::const_iterator itRaw = itRawDigis->begin(); 
     int strip =0, totADC=0;

     for(;itRaw != itRawDigis->end(); ++itRaw, ++strip){  /// loop over strips
       
       float adc = itRaw->adc();
       h1ADC_vs_strip_->Fill(strip,adc); /// adc vs strip



       totADC+= adc;
       
       if(strip%127 ==0){
	 if(totADC!= 0){
	   totADC =0;
	   
	   NumResAPVs++;

	 }
       }
       
     } ///strip loop ends
   
     
   }  /// module loop

       ///std::cout<< " napvs  : " << NumResAPVs << std::endl;

   h1NumbadAPVsRes_->Fill(NumResAPVs); /// for all modules



} /// analyzer loop



// ------------ method called once each job just after ending the event loop  ------------
void 
SiStripBaselineValidator::endJob() {

    if (!outputFile_.empty() && createOutputFile_) {
       dbe->save(outputFile_);
    }  
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripBaselineValidator);

