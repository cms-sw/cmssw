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
  SiPixelPhase1Base(iConfig)
{
  srcToken_ = consumes<edm::DetSetVector<PixelDigi>>(iConfig.getParameter<edm::InputTag>("src"));
} 

void SiPixelPhase1Digis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::Handle<edm::DetSetVector<PixelDigi>> input;
  iEvent.getByToken(srcToken_, input);
  if (!input.isValid()) return; 

  edm::DetSetVector<PixelDigi>::const_iterator it;
  for (it = input->begin(); it != input->end(); ++it) {
    for(PixelDigi const& digi : *it) {
      histo[ADC].fill((double) digi.adc(), DetId(it->detId()), &iEvent);
      histo[MAP].fill(DetId(it->detId()), &iEvent, digi.column(), digi.row()); 
      histo[NDIGIS    ].fill(DetId(it->detId()), &iEvent); // count
      histo[NDIGIS_FED].fill(DetId(it->detId()), &iEvent); 
    }
    histo[DEBUG].fill(geometryInterface.extract(geometryInterface.intern("PXLadder"), DetId(it->detId())), DetId(it->detId()));
  }
  histo[EVENT].fill(DetId(0), &iEvent);
  histo[NDIGIS    ].executePerEventHarvesting(); // TODO: move to Base?
  histo[NDIGIS_FED].executePerEventHarvesting(); 
}

DEFINE_FWK_MODULE(SiPixelPhase1Digis);

