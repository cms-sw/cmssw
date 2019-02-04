// Original Author:  Ivan Amos Cali
//         Created:  Mon Jul 28 14:10:52 CEST 2008
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
#include "DQMServices/Core/interface/DQMStore.h"


//ROOT inclusion
#include "TH1F.h"
#include "TH2F.h"
#include "TString.h"
#include <cassert>
#include <fstream>

class TFile;



using namespace edm;
using namespace std;

SiStripBaselineValidator::SiStripBaselineValidator(const edm::ParameterSet& conf){

  srcProcessedRawDigi_ =  conf.getParameter<edm::InputTag>( "srcProcessedRawDigi" );
  moduleRawDigiToken_ = consumes<edm::DetSetVector<SiStripDigi> >(conf.getParameter<edm::InputTag>( "srcProcessedRawDigi" ) );
}

SiStripBaselineValidator::~SiStripBaselineValidator()
{
}

void SiStripBaselineValidator::bookHistograms(DQMStore::IBooker & ibooker, const edm::Run & run, const edm::EventSetup & es)
{
  ///Setting the DQM top directories
  ibooker.setCurrentFolder("SiStrip/BaselineValidator");
  
  h1NumbadAPVsRes_ = ibooker.book1D("ResAPVs",";#ResAPVs", 100, 1.0, 10001);
  ibooker.tag(h1NumbadAPVsRes_,1);
  
  h1ADC_vs_strip_ = ibooker.book2D("ADCvsAPVs",";ADCvsAPVs", 768,-0.5,767.5,  1023, -0.5, 1022.5);
  ibooker.tag(h1ADC_vs_strip_,2);
  
  return;
}

void SiStripBaselineValidator::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  edm::Handle< edm::DetSetVector<SiStripDigi> > moduleRawDigi;
  e.getByToken( moduleRawDigiToken_, moduleRawDigi );
  edm::DetSetVector<SiStripDigi>::const_iterator itRawDigis = moduleRawDigi->begin();
 
   int NumResAPVs=0;
   for (; itRawDigis != moduleRawDigi->end(); ++itRawDigis) {   ///loop over modules
     

     edm::DetSet<SiStripDigi>::const_iterator itRaw = itRawDigis->begin(); 
     int strip =0, totStripAPV=0, apv=0,prevapv=itRaw->strip()/128;

     for(;itRaw != itRawDigis->end(); ++itRaw){  /// loop over strips
       
       strip=itRaw->strip();
       apv=strip/128;
       float adc = itRaw->adc();
       h1ADC_vs_strip_->Fill(strip,adc); /// adc vs strip

       if(prevapv!=apv){
         if(totStripAPV>64){
           NumResAPVs++;
         }
         prevapv=apv;
         totStripAPV=0;
       }
       if(adc>0) ++totStripAPV;

       
     } ///strip loop ends
     if(totStripAPV>64){
       NumResAPVs++;
     }   
     
   }  /// module loop

       ///std::cout<< " napvs  : " << NumResAPVs << std::endl;

   h1NumbadAPVsRes_->Fill(NumResAPVs); /// for all modules



} /// analyzer loop

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripBaselineValidator);

