#ifndef EcalDQMonitor_H
#define EcalDQMonitor_H

#include <string>
#include <vector>

#include "DQWorker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace edm
{
  class ParameterSet;
  class Run;
  class LuminosityBlock;
  class EventSetup;
}

namespace ecaldqm
{
  class EcalDQMonitor {
  public:
    EcalDQMonitor(edm::ParameterSet const&);
    virtual ~EcalDQMonitor();

    static void fillDescriptions(edm::ParameterSetDescription&);

  protected:
    void ecaldqmGetSetupObjects(edm::EventSetup const&);
    template<typename Booker> void ecaldqmBookHistograms(Booker&);
    void ecaldqmReleaseHistograms();
    void ecaldqmBeginRun(edm::Run const&, edm::EventSetup const&);
    void ecaldqmEndRun(edm::Run const&, edm::EventSetup const&);
    void ecaldqmBeginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
    void ecaldqmEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

    template<typename FuncOnWorker> void executeOnWorkers_(FuncOnWorker, std::string const&, std::string const& = "", int = 1); // loop over workers and capture exceptions
 
    std::vector<DQWorker*> workers_;
    std::string const moduleName_;
    int const verbosity_;
  };

  template<typename Booker>
    void
    EcalDQMonitor::ecaldqmBookHistograms(Booker& _booker)
    {
      executeOnWorkers_([&_booker](ecaldqm::DQWorker* worker){
          worker->releaseMEs();
          worker->bookMEs(_booker);
        }, "bookMEs", "Booking MEs");
    }

  template<typename FuncOnWorker>
    void
    EcalDQMonitor::executeOnWorkers_(FuncOnWorker _func, std::string const& _context, std::string const& _message/* = ""*/, int _verbThreshold/* = 1*/)
    {
      std::for_each(workers_.begin(), workers_.end(), [&](DQWorker* worker){
          if(verbosity_ > _verbThreshold && _message != "") edm::LogInfo("EcalDQM") << moduleName_ << ": " << _message << " @ " << worker->getName();
          try{
            _func(worker);
          }
          catch(std::exception&){
            edm::LogError("EcalDQM") << moduleName_ << ": Exception in " << _context << " @ " << worker->getName();
            throw;
          }
        });
    }
}

#endif
