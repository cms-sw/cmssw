// -*- C++ -*-
//
// Package:    SiPixelPhase1Digis
// Class:      SiPixelPhase1Digis
//

// Original Author: Marcel Schneider

#include "DQM/SiPixelPhase1Digis/interface/SiPixelPhase1Digis.h"

// C++ stuff
#include <iostream>

// CMSSW stuff
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// DQM Stuff
#include "DQMServices/Core/interface/MonitorElement.h"

SiPixelPhase1Digis::SiPixelPhase1Digis(const edm::ParameterSet& iConfig) :
  src_(iConfig.getParameter<edm::InputTag>("src")),
  topFolderName_(iConfig.getParameter<std::string>("TopFolderName"))
{
  srcToken_ = consumes<edm::DetSetVector<PixelDigi>>(iConfig.getParameter<edm::InputTag>("src"));
}

SiPixelPhase1Digis::~SiPixelPhase1Digis() {


}

void SiPixelPhase1Digis::dqmBeginRun(const edm::Run& r, const edm::EventSetup& iSetup) {
  std::cout << "++++ Begin run.\n";
}


void SiPixelPhase1Digis::bookHistograms(DQMStore::IBooker & iBooker, edm::Run const &, const edm::EventSetup & iSetup){
  iBooker.setCurrentFolder(topFolderName_);
  std::cout << "+++++ Booking.\n";
  alladcs = iBooker.book1D("alladcs","All ADC Values",100,0.,300.);

}

void SiPixelPhase1Digis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::Handle<edm::DetSetVector<PixelDigi>> input;
  iEvent.getByToken(srcToken_, input);
  if (!input.isValid()) return; 

  std::cout << "+++ Data valid.\n";
  
  edm::DetSetVector<PixelDigi>::const_iterator it;
  for (it = input->begin(); it != input->end(); ++it) {
    for(PixelDigi const& digi : *it) {
      alladcs->Fill(digi.adc());
    }
  }
}

DEFINE_FWK_MODULE(SiPixelPhase1Digis);

