#ifndef SiStripThresholdReader_H
#define SiStripThresholdReader_H

// system include files
//#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "CondFormats/DataRecord/interface/SiStripThresholdRcd.h"

#include <iostream>
#include <cstdio>
#include <sys/time.h>

class SiStripThresholdReader : public edm::EDAnalyzer {
public:
  explicit SiStripThresholdReader(const edm::ParameterSet&);
  ~SiStripThresholdReader() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  uint32_t printdebug_;
  edm::ESGetToken<SiStripThreshold, SiStripThresholdRcd> thresholdToken_;
};
#endif
