#ifndef CalibTracker_SiStripESProducer_SiStripFedCablingManipulator_h
#define CalibTracker_SiStripESProducer_SiStripFedCablingManipulator_h

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <string>

class SiStripFedCablingManipulator : public edm::EDAnalyzer {
public:
  explicit SiStripFedCablingManipulator(const edm::ParameterSet& iConfig);
  ~SiStripFedCablingManipulator() override;
  void analyze(const edm::Event& e, const edm::EventSetup& es) override{};

  void endRun(const edm::Run& run, const edm::EventSetup& es) override;

private:
  std::unique_ptr<SiStripFedCabling> manipulate(const SiStripFedCabling&);

  edm::ParameterSet iConfig_;
  edm::ESGetToken<SiStripFedCabling, SiStripFedCablingRcd> fedCablingToken_;
};

#endif
