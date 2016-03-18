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
  SiPixelPhase1Base(iConfig, MAX_HIST),
  src_(iConfig.getParameter<edm::InputTag>("src"))
{
  histo[ADC].setName("adc")
    .setTitle("Digi ADC values")
    .setXlabel("adc readout")
    .setDimensions(1);
  histo[ADC].addSpec()
    .groupBy("P1PXBBarrel/P1PXBHalfBarrel/P1PXBLayer/P1PXBLadder")
    .save();
  histo[ADC].addSpec()
    .groupBy("P1PXECEndcap/P1PXECHalfCylinder/P1PXECHalfDisk/P1PXECBlade")
    .save();

  histo[NDIGIS].setName("ndigis")
    .setTitle("Number of Digis")
    .setXlabel("#digis")
    .setDimensions(1);
  histo[NDIGIS].addSpec()
    .groupBy("P1PXBBarrel/P1PXBHalfBarrel/P1PXBLayer/P1PXBLadder")
    .save()
    .reduce("MEAN")
    .groupBy("P1PXBBarrel/P1PXBHalfBarrel/P1PXBLayer", "EXTEND_X")
    .save()
    .groupBy("P1PXBBarrel/P1PXBHalfBarrel", "EXTEND_X")
    .save()
    .groupBy("P1PXBBarrel", "EXTEND_X")
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
    int ndigis = 0;
    for(PixelDigi const& digi : *it) {
      histo[ADC].fill((double) digi.adc(), DetId(it->detId()), &iEvent, digi.column(), digi.row());
      ndigis++;
    }
    histo[NDIGIS].fill((double) ndigis, DetId(it->detId()), &iEvent);
  }
}

typedef SiPixelPhase1Analyzer<SiPixelPhase1Digis> SiPixelPhase1DigisAnalyzer;
typedef SiPixelPhase1Harvester<SiPixelPhase1Digis> SiPixelPhase1DigisHarvester;
DEFINE_FWK_MODULE(SiPixelPhase1DigisAnalyzer);
DEFINE_FWK_MODULE(SiPixelPhase1DigisHarvester);

