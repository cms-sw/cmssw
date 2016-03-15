#ifndef SiPixelPhase1Digis_h // Can we use #pagma once?
#define SiPixelPhase1Digis_h
// -*- C++ -*-
//
// Package:     SiPixelPhase1Digis
// Class  :     SiPixelPhase1Digis
// 

// Original Author: Marcel Schneider

// DQM Stuff
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

// Input data stuff
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

// PixelDQM Framework
#include "DQM/SiPixelPhase1Common/interface/HistogramManager.h"

class SiPixelPhase1Digis : public DQMEDAnalyzer {

  public:
  explicit SiPixelPhase1Digis(const edm::ParameterSet& conf);
  ~SiPixelPhase1Digis();

  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void dqmBeginRun(const edm::Run&, edm::EventSetup const&) override;
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  private:
  edm::InputTag src_;
  edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> srcToken_;

  HistogramManager histoman;

};

#endif
