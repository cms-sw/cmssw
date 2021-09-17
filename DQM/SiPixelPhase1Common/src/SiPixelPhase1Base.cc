// -*- C++ -*-
//
// Class:      SiPixelPhase1Base
//
// Implementations of the class
//
// Original Author: Yi-Mu "Enoch" Chen

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"

// Constructor requires manually looping the trigger flag settings
// Since constructor of GenericTriggerEventFlag requires
// EDConsumerBase class protected member calls
SiPixelPhase1Base::SiPixelPhase1Base(const edm::ParameterSet& iConfig)
    : DQMEDAnalyzer(), HistogramManagerHolder(iConfig, consumesCollector()) {
  // Flags will default to empty vector if not specified in configuration file
  auto flags = iConfig.getUntrackedParameter<edm::VParameterSet>("triggerflags", {});

  for (auto& flag : flags) {
    triggerlist.emplace_back(new GenericTriggerEventFlag(flag, consumesCollector(), *this));
  }
}

// Booking histograms as required by the DQM
void SiPixelPhase1Base::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const& run, edm::EventSetup const& iSetup) {
  for (HistogramManager& histoman : histo) {
    histoman.book(iBooker, iSetup);
  }

  // Running trigger flag initialization (per run)
  for (auto& trigger : triggerlist) {
    if (trigger->on()) {
      trigger->initRun(run, iSetup);
    }
  }
}

// trigger checking function
bool SiPixelPhase1Base::checktrigger(const edm::Event& iEvent,
                                     const edm::EventSetup& iSetup,
                                     const unsigned trgidx) const {
  //true if no trigger, MC, off, or accepted

  return triggerlist.empty() || !iEvent.isRealData() || !triggerlist.at(trgidx)->on() ||
         triggerlist.at(trgidx)->accept(iEvent, iSetup);
}
