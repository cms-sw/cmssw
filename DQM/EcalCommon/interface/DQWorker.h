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
    DQWorker(const edm::ParameterSet&, std::string const&);
    virtual ~DQWorker();

    virtual void beginRun(const edm::Run &, const edm::EventSetup &){};
    virtual void endRun(const edm::Run &, const edm::EventSetup &){};

    virtual void beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &){};
    virtual void endLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &){};

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

    static std::map<std::string, std::vector<MEData> > meData;
    // needs to be declared in each derived class
    static void setMEData(std::vector<MEData>&);

  protected :
    void meSet_(unsigned, edm::ParameterSet const&);
    MESet* createMESet_(MEData const&) const;

    std::string name_;
    std::vector<MESet*> MEs_; // [nMESets]
    bool initialized_;

    int verbosity_;
  };



  typedef DQWorker* (*WorkerFactory)(const edm::ParameterSet&);

  // template of WorkerFactory instance
  template<class W>
    DQWorker* 
    workerFactory(const edm::ParameterSet& _params)
    {
      return new W(_params);
    }

  // to be instantiated after the implementation of each worker module
  class SetWorker {
  public:
    template <class W> SetWorker(const std::string& _name, W*){
      workerFactories_[_name] = workerFactory<W>;

      std::vector<MEData>& data(DQWorker::meData[_name]);
      data.clear();
      data.resize(W::nMESets);
      W::setMEData(data);
    }
    static WorkerFactory findFactory(const std::string&);
  private:
    static std::map<std::string, WorkerFactory> workerFactories_;
  };

}

#define DEFINE_ECALDQM_WORKER(TYPE) \
  TYPE *p##TYPE(0); SetWorker TYPE##Instance(#TYPE, p##TYPE)

#endif
