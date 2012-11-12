#ifndef DQWorker_H
#define DQWorker_H

#include <string>
#include <vector>
#include <map>

#include "DQM/EcalCommon/interface/MESet.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm{
  class Run;
  class LuminosityBlock;
  class Event;
  class EventSetup;
}
class DQMStore;

namespace ecaldqm{

  class DQWorker {
  public :
    DQWorker(edm::ParameterSet const&, edm::ParameterSet const&, std::string const&);
    virtual ~DQWorker();

    virtual void beginRun(edm::Run const&, edm::EventSetup const&){};
    virtual void endRun(edm::Run const&, edm::EventSetup const&){};

    virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&){};
    virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&){};

    virtual void bookMEs();

    virtual void reset();
    virtual void initialize();
    virtual bool isInitialized() { return initialized_; }
    virtual void setVerbosity(int _verbosity) { verbosity_ = _verbosity; }

    std::string const& getName() const { return name_; }

    static bool online;
    static time_t now;
    static edm::RunNumber_t iRun;
    static edm::LuminosityBlockNumber_t iLumi;
    static edm::EventNumber_t iEvt;

  protected :
    void print_(std::string const&, int = 0) const;

    std::string name_;
    MESetCollection MEs_;
    bool initialized_;

    int verbosity_;
  };



  typedef DQWorker* (*WorkerFactory)(edm::ParameterSet const&, edm::ParameterSet const&);

  // template of WorkerFactory instance
  template<class W>
    DQWorker* 
    workerFactory(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams)
    {
      return new W(_workerParams, _commonParams);
    }

  // to be instantiated after the implementation of each worker module
  class WorkerFactoryHelper {
  public:
    template <class W> WorkerFactoryHelper(const std::string& _name, W*){
      workerFactories_[_name] = workerFactory<W>;
    }
    static WorkerFactory findFactory(const std::string&);
  private:
    static std::map<std::string, WorkerFactory> workerFactories_;
  };

}

#define DEFINE_ECALDQM_WORKER(TYPE) \
  TYPE *p##TYPE(0); WorkerFactoryHelper TYPE##Instance(#TYPE, p##TYPE)

#endif
