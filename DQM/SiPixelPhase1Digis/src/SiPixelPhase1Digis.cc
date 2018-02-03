// -*- C++ -*-
//
// Package:    SiPixelPhase1Digis
// Class:      SiPixelPhase1Digis
//

// Original Author: Marcel Schneider

// C++ stuff
#include <iostream>

// CMSSW stuff
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// DQM Stuff
#include "DQMServices/Core/interface/MonitorElement.h"

// Input data stuff
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

// PixelDQM Framework
#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"

namespace {

class SiPixelPhase1Digis final : public SiPixelPhase1Base {
  // List of quantities to be plotted. 
  enum {
    ADC, // digi ADC readouts
    NDIGIS, // number of digis per event and module
    NDIGISINCLUSIVE, //Total number of digis in BPix and FPix
    NDIGIS_FED, // number of digis per event and FED
    NDIGIS_FEDtrend, // number of digis per event and FED 
    EVENT, // event frequency
    MAP, // digi hitmap per module
    OCCUPANCY, // like map but coarser

    MAX_HIST // a sentinel that gives the number of quantities (not a plot).
  };
  public:
  explicit SiPixelPhase1Digis(const edm::ParameterSet& conf);

  void analyze(const edm::Event&, const edm::EventSetup&) override ;

  private:
  edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> srcToken_;

};

SiPixelPhase1Digis::SiPixelPhase1Digis(const edm::ParameterSet& iConfig) :
  SiPixelPhase1Base(iConfig)
{
  srcToken_ = consumes<edm::DetSetVector<PixelDigi>>(iConfig.getParameter<edm::InputTag>("src"));
} 

void SiPixelPhase1Digis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if( !checktrigger(iEvent,iSetup,DCS) ) return;

  edm::Handle<edm::DetSetVector<PixelDigi>> input;
  iEvent.getByToken(srcToken_, input);
  if (!input.isValid()) return; 
  bool hasDigis=false;

  edm::DetSetVector<PixelDigi>::const_iterator it;
  for (it = input->begin(); it != input->end(); ++it) {
    for(PixelDigi const& digi : *it) {
      hasDigis=true;
      histo[ADC].fill((double) digi.adc(), DetId(it->detId()), &iEvent, digi.column(), digi.row());
      histo[MAP].fill(DetId(it->detId()), &iEvent, digi.column(), digi.row()); 
      histo[OCCUPANCY].fill(DetId(it->detId()), &iEvent, digi.column(), digi.row()); 
      histo[NDIGIS    ].fill(DetId(it->detId()), &iEvent); // count
      histo[NDIGISINCLUSIVE].fill(DetId(it->detId()), &iEvent); // count
      histo[NDIGIS_FED].fill(DetId(it->detId()), &iEvent); 
      histo[NDIGIS_FEDtrend].fill(DetId(it->detId()), &iEvent);  
    }
  }
  if (hasDigis) histo[EVENT].fill(DetId(0), &iEvent);
  histo[NDIGIS    ].executePerEventHarvesting(&iEvent);
  histo[NDIGISINCLUSIVE].executePerEventHarvesting(&iEvent);
  histo[NDIGIS_FED].executePerEventHarvesting(&iEvent); 
  histo[NDIGIS_FEDtrend].executePerEventHarvesting(&iEvent);
}

} //namespace

DEFINE_FWK_MODULE(SiPixelPhase1Digis);

