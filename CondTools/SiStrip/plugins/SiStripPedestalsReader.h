#ifndef SiStripPedestalsReader_H
#define SiStripPedestalsReader_H

// system include files
//#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"

#include <iostream>
#include <cstdio>
#include <sys/time.h>

class SiStripPedestalsReader : public edm::one::EDAnalyzer<> {
public:
  explicit SiStripPedestalsReader(const edm::ParameterSet&);
  ~SiStripPedestalsReader() override = default;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  uint32_t printdebug_;
  edm::ESGetToken<SiStripPedestals, SiStripPedestalsRcd> pedestalsToken_;
};
#endif
