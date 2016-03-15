#ifndef SiPixel_BasicFrameworkTest_h
#define SiPixel_BasicFrameworkTest_h

// Original Author:  Marcel Schneider

// system includes
#include <iostream>

// core framework functionality
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// PixelDQM framework
#include "DQM/SiPixelPhase1Common/interface/HistogramManager.h"

class BasicFrameworkTest : public edm::EDAnalyzer {

  public:
  BasicFrameworkTest(const edm::ParameterSet&);
  virtual ~BasicFrameworkTest() {};

  virtual void beginRun(const edm::Run&,   const edm::EventSetup&) override;
  virtual void analyze (const edm::Event&, const edm::EventSetup&) {};

  private:
  HistogramManager histoman;

};


#endif
