#ifndef EcalCondDBReader_H
#define EcalCondDBReader_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "DQM/EcalCommon/interface/MESet.h"
#include "DBReaderWorkers.h"

#include "TFormula.h"

class EcalCondDBReader : public edm::EDAnalyzer {
 public:
  EcalCondDBReader(edm::ParameterSet const&);
  ~EcalCondDBReader();

 private:
  void analyze(edm::Event const&, edm::EventSetup const&);

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
  MonRunIOV monIOV_;
  ecaldqm::DBReaderWorker* worker_;
  std::string formula_;
  ecaldqm::MESet* meSet_;

  int verbosity_;
  bool executed_;
};

#endif
