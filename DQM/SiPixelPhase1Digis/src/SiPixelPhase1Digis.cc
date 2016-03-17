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
  SiPixelPhase1Base(iConfig),
  src_(iConfig.getParameter<edm::InputTag>("src"))
{
  histoman.setName("adc");
  histoman.setTitle("Digi ADC values");
  histoman.setXlabel("adc readout");
  histoman.setDimensions(1);
  histoman.addSpec()
    .groupBy("P1PXBBarrel/P1PXBHalfBarrel/P1PXBLayer/P1PXBLadder")
    .save();
  histoman.addSpec()
    .groupBy("P1PXECEndcap/P1PXECHalfCylinder/P1PXECHalfDisk/P1PXECBlade")
    .save();
} 

template<class Consumer>
void SiPixelPhase1Digis::registerConsumes(const edm::ParameterSet& iConfig, Consumer& c) {
  srcToken_ = c.template consumes<edm::DetSetVector<PixelDigi>>(iConfig.getParameter<edm::InputTag>("src"));
}
  

void SiPixelPhase1Digis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::Handle<edm::DetSetVector<PixelDigi>> input;
  iEvent.getByToken(srcToken_, input);
  if (!input.isValid()) return; 

  std::cout << "+++ Data valid.\n";
  
  edm::DetSetVector<PixelDigi>::const_iterator it;
  for (it = input->begin(); it != input->end(); ++it) {
    for(PixelDigi const& digi : *it) {
      histoman.fill((double) digi.adc(), DetId(it->detId()), &iEvent, digi.column(), digi.row());
    }
  }
}

typedef SiPixelPhase1Analyzer<SiPixelPhase1Digis> SiPixelPhase1DigisAnalyzer;
typedef SiPixelPhase1Harvester<SiPixelPhase1Digis> SiPixelPhase1DigisHarvester;
DEFINE_FWK_MODULE(SiPixelPhase1DigisAnalyzer);
DEFINE_FWK_MODULE(SiPixelPhase1DigisHarvester);

