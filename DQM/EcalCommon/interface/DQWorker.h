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
    std::vector<MESet*> const& getMEs() const { return MEs_; }

    enum MESets {
      nMESets
    };

    static std::map<std::string, std::map<std::string, unsigned> > meOrderingMaps;
    // needs to be declared in each derived class
    static void setMEOrdering(std::map<std::string, unsigned>&);

  protected :
    MESet* createMESet_(std::string const&, edm::ParameterSet const&) const;
    void print_(std::string const&, int = 0) const;

    std::string name_;
    std::vector<MESet*> MEs_; // [nMESets]
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

      std::map<std::string, unsigned>& oMap(DQWorker::meOrderingMaps[_name]);
      W::setMEOrdering(oMap);
    }
    static WorkerFactory findFactory(const std::string&);
  private:
    static std::map<std::string, WorkerFactory> workerFactories_;
  };

}

#define DEFINE_ECALDQM_WORKER(TYPE) \
  TYPE *p##TYPE(0); WorkerFactoryHelper TYPE##Instance(#TYPE, p##TYPE)

#endif
