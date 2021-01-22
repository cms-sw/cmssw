#ifndef SiStripSummaryReader_H
#define SiStripSummaryReader_H

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

#include "CondFormats/DataRecord/interface/SiStripSummaryRcd.h"
class SiStripSummary;

class SiStripSummaryReader : public edm::EDAnalyzer {
public:
  explicit SiStripSummaryReader(const edm::ParameterSet&);
  ~SiStripSummaryReader() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  uint32_t printdebug_;
  edm::ESGetToken<SiStripSummary, SiStripSummaryRcd> summaryToken_;
};
#endif
