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

#endif
