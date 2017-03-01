#ifndef CalibTracker_SiStripESProducer_SiStripFedCablingManipulator_h
#define CalibTracker_SiStripESProducer_SiStripFedCablingManipulator_h

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <string>

class SiStripFedCabling;

class SiStripFedCablingManipulator : public edm::EDAnalyzer {

public:

  explicit SiStripFedCablingManipulator(const edm::ParameterSet& iConfig);
  ~SiStripFedCablingManipulator();
  void analyze(const edm::Event& e, const edm::EventSetup&es){};

  void endRun(const edm::Run & run, const edm::EventSetup & es);

 private:

  void manipulate(const SiStripFedCabling*,SiStripFedCabling*&);

  edm::ParameterSet iConfig_;
};

#endif
