#ifndef SiPixelPhase1Digis_h // Can we use #pragma once?
#define SiPixelPhase1Digis_h
// -*- C++ -*-
//
// Package:     SiPixelPhase1Digis
// Class  :     SiPixelPhase1Digis
// 

// Original Author: Marcel Schneider

// Input data stuff
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

// PixelDQM Framework
#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"

class SiPixelPhase1Digis : public SiPixelPhase1Base {
  // List of quantities to be plotted. 
  enum {
    ADC, // digi ADC readouts
    NDIGIS, // number of digis per event and module
    NDIGIS_FED, // number of digis per event and FED
    NDIGIS_FEDtrend, // number of digis per event and FED 
    EVENT, // event frequency
    MAP, // digi hitmap per module

    MAX_HIST // a sentinel that gives the number of quantities (not a plot).
  };
  public:
  explicit SiPixelPhase1Digis(const edm::ParameterSet& conf);

  void analyze(const edm::Event&, const edm::EventSetup&) ;

  private:
  edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> srcToken_;

};

class SiPixelPhase1DigisHarvester : public SiPixelPhase1Harvester {
  enum {
    ADC,
    NDIGIS,
    NDIGIS_FED, 
    NDIGIS_FEDtrend,
    EVENT,
    MAP,

    MAX_HIST
  };
  public:
  explicit SiPixelPhase1DigisHarvester(const edm::ParameterSet& conf);

};

#endif
