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
  SiPixelPhase1Base(iConfig, MAX_HIST)
{
  histo[ADC].setName("adc")
    .setTitle("Digi ADC values")
    .setXlabel("adc readout")
    .setRange(300, 0, 300)
    .setDimensions(1);
  histo[ADC].addSpec()
    .groupBy(histo[ADC].defaultGrouping())
    .saveAll();
  histo[ADC].addSpec()
    .groupBy("BX")
    .reduce("COUNT")
    .groupBy("", "EXTEND_X")
    .save();
  histo[ADC].addSpec()
    .groupBy(histo[ADC].defaultGrouping() + "/row/col")
    .reduce("COUNT")
    .groupBy(histo[ADC].defaultGrouping() + "/row", "EXTEND_X")
    .groupBy(histo[ADC].defaultGrouping(), "EXTEND_Y")
    .saveAll();

  histo[MAP].setName("hitmap")
    .setTitle("Position of digis on module")
    .setXlabel("col")
    .setYlabel("row")
    .setRange(200, 0, 200)
    .setDimensions(2)
    .addSpec()
      .groupBy("P1PXBBarrel|P1PXECEndcap/P1PXBHalfBarrel|P1PXECHalfCylinder/P1PXBLayer|P1PXECHalfDisk/P1PXBLadder|P1PXECBlade/DetUnit")
      .save()
      .groupBy("P1PXBBarrel|P1PXECEndcap/P1PXBHalfBarrel|P1PXECHalfCylinder/P1PXBLayer|P1PXECHalfDisk/P1PXBLadder|P1PXECBlade", "EXTEND_X")
      .save()
      .groupBy("P1PXBBarrel|P1PXECEndcap/P1PXBHalfBarrel|P1PXECHalfCylinder/P1PXBLayer|P1PXECHalfDisk", "SUM")
      .saveAll();



  histo[NDIGIS].setName("ndigis")
    .setTitle("Number of Digis")
    .setXlabel("#digis")
    .setRange(30, 0, 30)
    .setDimensions(1);
  histo[NDIGIS].addSpec()
    .groupBy(histo[NDIGIS].defaultGrouping())
    .save()
    .reduce("MEAN")
    // TODO: find a way to express this with default. defaultGrouping(1) or sth.?
    .groupBy("P1PXBBarrel|P1PXECEndcap/P1PXBHalfBarrel|P1PXECHalfCylinder/P1PXBLayer|P1PXECHalfDisk", "EXTEND_X")
    .saveAll();


} 

template<class Consumer>
void SiPixelPhase1Digis::registerConsumes(const edm::ParameterSet& iConfig, Consumer& c) {
  srcToken_ = c.template consumes<edm::DetSetVector<PixelDigi>>(iConfig.getParameter<edm::InputTag>("src"));
}
  

void SiPixelPhase1Digis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::Handle<edm::DetSetVector<PixelDigi>> input;
  iEvent.getByToken(srcToken_, input);
  if (!input.isValid()) return; 

  edm::DetSetVector<PixelDigi>::const_iterator it;
  for (it = input->begin(); it != input->end(); ++it) {
    int ndigis = 0;
    for(PixelDigi const& digi : *it) {
      histo[ADC].fill((double) digi.adc(), DetId(it->detId()), &iEvent, digi.column(), digi.row());
      histo[MAP].fill((double) digi.column(), (double) digi.row(), DetId(it->detId())); 
      ndigis++;
    }
    histo[NDIGIS].fill((double) ndigis, DetId(it->detId()), &iEvent);
  }
}

typedef SiPixelPhase1Analyzer<SiPixelPhase1Digis> SiPixelPhase1DigisAnalyzer;
typedef SiPixelPhase1Harvester<SiPixelPhase1Digis> SiPixelPhase1DigisHarvester;
DEFINE_FWK_MODULE(SiPixelPhase1DigisAnalyzer);
DEFINE_FWK_MODULE(SiPixelPhase1DigisHarvester);

