#ifndef DBReaderWorkers_H
#define DBReaderWorkers_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"

#include "DataFormats/DetId/interface/DetId.h"

#include <map>

namespace ecaldqm
{
  class DBReaderWorker {
  public:
    DBReaderWorker(std::string const& _name, edm::ParameterSet const&) : name_(_name), verbosity_(0) {}
    virtual ~DBReaderWorker() {}

    virtual std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) = 0;

    void setVerbosity(int _v) { verbosity_ = _v; }

    std::string const& getName() const { return name_; }

  protected:
    std::string const name_;
    int verbosity_;
  };

  class CrystalConsistencyReader : public DBReaderWorker {
  public:
    CrystalConsistencyReader(edm::ParameterSet const& _ps) : DBReaderWorker("CrystalConsistencyReader", _ps) {}
    ~CrystalConsistencyReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class TTConsistencyReader : public DBReaderWorker {
  public:
    TTConsistencyReader(edm::ParameterSet const& _ps) : DBReaderWorker("TTConsistencyReader", _ps) {}
    ~TTConsistencyReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class MemChConsistencyReader : public DBReaderWorker {
  public:
    MemChConsistencyReader(edm::ParameterSet const& _ps) : DBReaderWorker("MemChConsistencyReader", _ps) {}
    ~MemChConsistencyReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class MemTTConsistencyReader : public DBReaderWorker {
  public:
    MemTTConsistencyReader(edm::ParameterSet const& _ps) : DBReaderWorker("MemTTConsistencyReader", _ps) {}
    ~MemTTConsistencyReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class LaserBlueReader : public DBReaderWorker {
  public:
    LaserBlueReader(edm::ParameterSet const& _ps) : DBReaderWorker("LaserBlueReader", _ps) {}
    ~LaserBlueReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class TimingLaserBlueCrystalReader : public DBReaderWorker {
  public:
    TimingLaserBlueCrystalReader(edm::ParameterSet const& _ps) : DBReaderWorker("TimingLaserBlueCrystalReader", _ps) {}
    ~TimingLaserBlueCrystalReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class PNBlueReader : public DBReaderWorker {
  public:
    PNBlueReader(edm::ParameterSet const& _ps) : DBReaderWorker("PNBlueReader", _ps) {}
    ~PNBlueReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class LaserGreenReader : public DBReaderWorker {
  public:
    LaserGreenReader(edm::ParameterSet const& _ps) : DBReaderWorker("LaserGreenReader", _ps) {}
    ~LaserGreenReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class TimingLaserGreenCrystalReader : public DBReaderWorker {
  public:
    TimingLaserGreenCrystalReader(edm::ParameterSet const& _ps) : DBReaderWorker("TimingLaserGreenCrystalReader", _ps) {}
    ~TimingLaserGreenCrystalReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class PNGreenReader : public DBReaderWorker {
  public:
    PNGreenReader(edm::ParameterSet const& _ps) : DBReaderWorker("PNGreenReader", _ps) {}
    ~PNGreenReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class LaserIRedReader : public DBReaderWorker {
  public:
    LaserIRedReader(edm::ParameterSet const& _ps) : DBReaderWorker("LaserIRedReader", _ps) {}
    ~LaserIRedReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class TimingLaserIRedCrystalReader : public DBReaderWorker {
  public:
    TimingLaserIRedCrystalReader(edm::ParameterSet const& _ps) : DBReaderWorker("TimingLaserIRedCrystalReader", _ps) {}
    ~TimingLaserIRedCrystalReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class PNIRedReader : public DBReaderWorker {
  public:
    PNIRedReader(edm::ParameterSet const& _ps) : DBReaderWorker("PNIRedReader", _ps) {}
    ~PNIRedReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class LaserRedReader : public DBReaderWorker {
  public:
    LaserRedReader(edm::ParameterSet const& _ps) : DBReaderWorker("LaserRedReader", _ps) {}
    ~LaserRedReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class TimingLaserRedCrystalReader : public DBReaderWorker {
  public:
    TimingLaserRedCrystalReader(edm::ParameterSet const& _ps) : DBReaderWorker("TimingLaserRedCrystalReader", _ps) {}
    ~TimingLaserRedCrystalReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class PNRedReader : public DBReaderWorker {
  public:
    PNRedReader(edm::ParameterSet const& _ps) : DBReaderWorker("PNRedReader", _ps) {}
    ~PNRedReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class PedestalsReader : public DBReaderWorker {
  public:
    PedestalsReader(edm::ParameterSet const& _ps) : DBReaderWorker("PedestalsReader", _ps) {}
    ~PedestalsReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class PNPedReader : public DBReaderWorker {
  public:
    PNPedReader(edm::ParameterSet const& _ps) : DBReaderWorker("PNPedReader", _ps) {}
    ~PNPedReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class PedestalsOnlineReader : public DBReaderWorker {
  public:
    PedestalsOnlineReader(edm::ParameterSet const& _ps) : DBReaderWorker("PedestalsOnlineReader", _ps) {}
    ~PedestalsOnlineReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class TestPulseReader : public DBReaderWorker {
  public:
    TestPulseReader(edm::ParameterSet const& _ps) : DBReaderWorker("TestPulseReader", _ps) {}
    ~TestPulseReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class PulseShapeReader : public DBReaderWorker {
  public:
    PulseShapeReader(edm::ParameterSet const& _ps) : DBReaderWorker("PulseShapeReader", _ps) {}
    ~PulseShapeReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class PNMGPAReader : public DBReaderWorker {
  public:
    PNMGPAReader(edm::ParameterSet const& _ps) : DBReaderWorker("PNMGPAReader", _ps) {}
    ~PNMGPAReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class TimingCrystalReader : public DBReaderWorker {
  public:
    TimingCrystalReader(edm::ParameterSet const& _ps) : DBReaderWorker("TimingCrystalReader", _ps) {}
    ~TimingCrystalReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class Led1Reader : public DBReaderWorker {
  public:
    Led1Reader(edm::ParameterSet const& _ps) : DBReaderWorker("Led1Reader", _ps) {}
    ~Led1Reader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class TimingLed1CrystalReader : public DBReaderWorker {
  public:
    TimingLed1CrystalReader(edm::ParameterSet const& _ps) : DBReaderWorker("TimingLed1CrystalReader", _ps) {}
    ~TimingLed1CrystalReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class Led2Reader : public DBReaderWorker {
  public:
    Led2Reader(edm::ParameterSet const& _ps) : DBReaderWorker("Led2Reader", _ps) {}
    ~Led2Reader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class TimingLed2CrystalReader : public DBReaderWorker {
  public:
    TimingLed2CrystalReader(edm::ParameterSet const& _ps) : DBReaderWorker("TimingLed2CrystalReader", _ps) {}
    ~TimingLed2CrystalReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
  
  class OccupancyReader : public DBReaderWorker {
  public:
    OccupancyReader(edm::ParameterSet const& _ps) : DBReaderWorker("OccupancyReader", _ps) {}
    ~OccupancyReader() {}

    std::map<DetId, double> run(EcalCondDBInterface*, MonRunIOV&, std::string const&) override;
  };
}

#endif
