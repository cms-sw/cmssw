#ifndef EcalCondDBReader_H
#define EcalCondDBReader_H

#include "DBReaderWorkers.h"
#include "DQM/EcalCommon/interface/MESet.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

class EcalCondDBReader : public DQMEDHarvester {
public:
  EcalCondDBReader(edm::ParameterSet const &);
  ~EcalCondDBReader() override;

private:
  void dqmEndRun(DQMStore::IBooker &, DQMStore::IGetter &, edm::Run const &, edm::EventSetup const &) override;

  EcalElectronicsMapping const *electronicsMap;
  void setElectronicsMap(edm::EventSetup const &);
  EcalElectronicsMapping const *GetElectronicsMap();
  ecaldqm::EcalDQMSetupObjects const getEcalDQMSetupObjects();

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

  EcalCondDBInterface *db_;
  MonRunIOV monIOV_;
  ecaldqm::DBReaderWorker *worker_;
  std::string formula_;
  ecaldqm::MESet *meSet_;

  int verbosity_;
  bool executed_;
};

#endif
