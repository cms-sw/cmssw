// -*- C++ -*-
//
// Package:   BeamSplash
// Class:     BeamSPlash
//
//
// Original Author:  Luca Malgeri

#include <memory>
#include <vector>
#include <map>
#include <set>

// user include files

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DPGAnalysis/Skims/interface/FilterOutScraping.h"


using namespace edm;
using namespace std;

FilterOutScraping::FilterOutScraping(const edm::ParameterSet& iConfig)
{
  
  applyfilter = iConfig.getUntrackedParameter<bool>("applyfilter",true);
  debugOn     = iConfig.getUntrackedParameter<bool>("debugOn",false);
  thresh =  iConfig.getUntrackedParameter<double>("thresh",0.2);
  numtrack = iConfig.getUntrackedParameter<unsigned int>("numtrack",10);
  tracks_ = iConfig.getUntrackedParameter<edm::InputTag>("src",edm::InputTag("generalTracks"));
}

FilterOutScraping::~FilterOutScraping()
{
}

bool FilterOutScraping::filter( edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  bool accepted = false;
  float fraction = 0;  
  // get GeneralTracks collection

  edm::Handle<reco::TrackCollection> tkRef;
  iEvent.getByLabel(tracks_,tkRef);    
  const reco::TrackCollection* tkColl = tkRef.product();

  //std::cout << "Total Number of Tracks " << tkColl->size() << std::endl;
  
  int numhighpurity=0;
  _trackQuality = reco::TrackBase::qualityByName("highPurity");

  if(tkColl->size()>numtrack){ 
    reco::TrackCollection::const_iterator itk = tkColl->begin();
    reco::TrackCollection::const_iterator itk_e = tkColl->end();
    for(;itk!=itk_e;++itk){
      // std::cout << "HighPurity?  " << itk->quality(_trackQuality) << std::endl;
      if(itk->quality(_trackQuality)) numhighpurity++;
    }
    fraction = (float)numhighpurity/(float)tkColl->size();
    if(fraction>thresh) accepted=true;
  }else{
    //if less than 10 Tracks accept the event anyway    
    accepted= true;
  }
  
  
  if (debugOn) {
    int ievt = iEvent.id().event();
    int irun = iEvent.id().run();
    int ils = iEvent.luminosityBlock();
    int bx = iEvent.bunchCrossing();
    
    std::cout << "FilterOutScraping_debug: Run " << irun << " Event " << ievt << " Lumi Block " << ils << " Bunch Crossing " << bx << " Fraction " << fraction << " NTracks " << tkColl->size() << " Accepted " << accepted << std::endl;
  }
 
  if (applyfilter)
    return accepted;
  else
    return true;

}

//define this as a plug-in
DEFINE_FWK_MODULE(FilterOutScraping);
