#ifndef DQWorker_H
#define DQWorker_H

#include <string>
#include <vector>
#include <map>

#include "DQM/EcalCommon/interface/MESet.h"

#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/EventID.h"

#include "tbb/concurrent_unordered_map.h"

namespace edm{
  class Run;
  class LuminosityBlock;
  class Event;
  class EventSetup;
  class ParameterSet;
  class ParameterSetDescription;
}

namespace ecaldqm{

  class WorkerFactoryStore;

  class DQWorker {
    friend class WorkerFactoryStore;

  private:
    struct Timestamp {
      time_t now;
      edm::RunNumber_t iRun;
      edm::LuminosityBlockNumber_t iLumi;
      edm::EventNumber_t iEvt;
      Timestamp() : now(0), iRun(0), iLumi(0), iEvt(0) {}
    };

  protected:
    void setVerbosity(int _verbosity) { verbosity_ = _verbosity; }
    void initialize(std::string const& _name, edm::ParameterSet const&);

    virtual void setME(edm::ParameterSet const&);
    virtual void setSource(edm::ParameterSet const&) {} // for clients
    virtual void setParams(edm::ParameterSet const&) {}

  public:
    DQWorker();
    virtual ~DQWorker();

    static void fillDescriptions(edm::ParameterSetDescription& _desc);

    virtual void beginRun(edm::Run const&, edm::EventSetup const&) {}
    virtual void endRun(edm::Run const&, edm::EventSetup const&) {}

    virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
    virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}

    virtual void bookMEs(DQMStore&);
    virtual void bookMEs(DQMStore::IBooker&);
    virtual void releaseMEs();

    void setTime(time_t _t) { timestamp_.now = _t; }
    void setRunNumber(edm::RunNumber_t _r) { timestamp_.iRun = _r; }
    void setLumiNumber(edm::LuminosityBlockNumber_t _l) { timestamp_.iLumi = _l; }
    void setEventNumber(edm::EventNumber_t _e) { timestamp_.iEvt = _e; }

    std::string const& getName() const { return name_; }
    bool onlineMode() const { return onlineMode_; } 

  protected:
    void print_(std::string const&, int = 0) const;

    std::string name_;
    MESetCollection MEs_;

    Timestamp timestamp_;
    int verbosity_;

    // common parameters
    bool onlineMode_;
    bool willConvertToEDM_;
  };


  typedef DQWorker* (*WorkerFactory)();

  // to be instantiated after the implementation of each worker module
  class WorkerFactoryStore {
  public:
    template<typename Worker>
    struct Registration {
      Registration(std::string const& _name){ WorkerFactoryStore::singleton()->registerFactory(_name, []() -> DQWorker* { return new Worker(); }); }
    };
    
    void registerFactory(std::string const& _name, WorkerFactory _f) { workerFactories_[_name] = _f; }
    DQWorker* getWorker(std::string const&, int, edm::ParameterSet const&, edm::ParameterSet const&) const;

    static WorkerFactoryStore* singleton();

  private:
    tbb::concurrent_unordered_map<std::string, WorkerFactory> workerFactories_;
  };

}

#define DEFINE_ECALDQM_WORKER(TYPE) \
  WorkerFactoryStore::Registration<TYPE> ecaldqm##TYPE##Registration(#TYPE)

#endif
