#ifndef EcalCondDBWriter_H
#define EcalCondDBWriter_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DBWriterWorkers.h"

class EcalCondDBInterface;

class EcalCondDBWriter : public edm::EDAnalyzer {
 public:
  EcalCondDBWriter(edm::ParameterSet const&);
  ~EcalCondDBWriter();

  static void fillDescriptions(edm::ConfigurationDescriptions&);

 private:
  void analyze(edm::Event const&, edm::EventSetup const&) {}
  void endJob();

  // DON'T CHANGE - ORDER MATTERS IN DB
  enum Tasks {
    Integrity = 0,
    Cosmic = 1,
    Laser = 2,
    Pedestal = 3,
    Presample = 4,
    TestPulse = 5,
    BeamCalo = 6,
    BeamHodo = 7,
    TriggerPrimitives = 8,
    Cluster = 9,
    Timing = 10,
    Led = 11,
    RawData = 12,
    Occupancy = 13,
    nTasks = 14
  };

  EcalCondDBInterface* db_;
  std::string location_;
  std::string runType_;
  std::string runGeneralTag_;
  std::string monRunGeneralTag_;
  std::vector<std::string> inputRootFiles_;
  ecaldqm::DBWriterWorker* workers_[nTasks];
  ecaldqm::SummaryWriter summaryWriter_;

  int verbosity_;
  bool executed_;
};

#endif
