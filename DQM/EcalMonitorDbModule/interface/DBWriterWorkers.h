#ifndef DBWriterWorkers_H
#define DBWriterWorkers_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/EcalCommon/interface/MESet.h"

#include "OnlineDB/EcalCondDB/interface/MonRunDat.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include <map>

class DQMStore;

namespace ecaldqm {

  class DBWriterWorker {
  public:
    DBWriterWorker(std::string const&, edm::ParameterSet const&);
    virtual ~DBWriterWorker() {}

    virtual void retrieveSource(DQMStore const&);
    virtual bool run(EcalCondDBInterface*, MonRunIOV&) = 0;

    bool runsOn(std::string const& _runType) const { return runTypes_.find(_runType) != runTypes_.end(); }

    void setVerbosity(int _v) { verbosity_ = _v; }

    std::string const& getName() const { return name_; }
    bool isActive() const { return active_; }

  protected:
    std::string const name_;
    std::set<std::string> runTypes_;
    MESetCollection source_;
    bool active_;
    int verbosity_;
  };

  class IntegrityWriter : public DBWriterWorker {
  public:
    IntegrityWriter(edm::ParameterSet const& _ps) : DBWriterWorker("Integrity", _ps) {}
    ~IntegrityWriter() {}

    bool run(EcalCondDBInterface*, MonRunIOV&) override;
  };

  class LaserWriter : public DBWriterWorker {
  public:
    LaserWriter(edm::ParameterSet const&);
    ~LaserWriter() {}

    bool run(EcalCondDBInterface*, MonRunIOV&) override;

  private:
    std::map<int, unsigned> wlToME_;
  };

  class PedestalWriter : public DBWriterWorker {
  public:
    PedestalWriter(edm::ParameterSet const&);
    ~PedestalWriter() {}

    bool run(EcalCondDBInterface*, MonRunIOV&) override;

  private:
    std::map<int, unsigned> gainToME_;
    std::map<int, unsigned> pnGainToME_;
  };

  class PresampleWriter : public DBWriterWorker {
  public:
    PresampleWriter(edm::ParameterSet const& _ps) : DBWriterWorker("Presample", _ps) {}
    ~PresampleWriter() {}

    bool run(EcalCondDBInterface*, MonRunIOV&) override;
  };

  class TestPulseWriter : public DBWriterWorker {
  public:
    TestPulseWriter(edm::ParameterSet const&);
    ~TestPulseWriter() {}

    bool run(EcalCondDBInterface*, MonRunIOV&) override;

  private:
    std::map<int, unsigned> gainToME_;
    std::map<int, unsigned> pnGainToME_;
  };

  class TimingWriter : public DBWriterWorker {
  public:
    TimingWriter(edm::ParameterSet const& _ps) : DBWriterWorker("Timing", _ps) {}
    ~TimingWriter() {}

    bool run(EcalCondDBInterface*, MonRunIOV&) override;
  };

  class LedWriter : public DBWriterWorker {
  public:
    LedWriter(edm::ParameterSet const&);
    ~LedWriter() {}

    bool run(EcalCondDBInterface*, MonRunIOV&) override;

  private:
    std::map<int, unsigned> wlToME_;
  };

  class OccupancyWriter : public DBWriterWorker {
  public:
    OccupancyWriter(edm::ParameterSet const& _ps) : DBWriterWorker("Occupancy", _ps) {}
    ~OccupancyWriter() {}

    bool run(EcalCondDBInterface*, MonRunIOV&) override;
  };

  class SummaryWriter : public DBWriterWorker {
  public:
    SummaryWriter(edm::ParameterSet const& _ps) : DBWriterWorker("Summary", _ps), taskList_(0), outcome_(0), processedEvents_(0) {}
    ~SummaryWriter() {}

    void setTaskList(int _list) { taskList_ = _list; }
    void setOutcome(int _outcome) { outcome_ = _outcome; }
    void setProcessedEvents(unsigned _n) { processedEvents_ = _n; }
    bool run(EcalCondDBInterface*, MonRunIOV&) override;

  private:
    int taskList_;
    int outcome_;
    unsigned processedEvents_;
  };
}

#endif
