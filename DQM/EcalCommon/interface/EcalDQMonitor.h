#ifndef EcalDQMonitor_H
#define EcalDQMonitor_H

#include <string>
#include <vector>

#include "DQWorker.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace edm {
  class ParameterSet;
  class Run;
  class LuminosityBlock;
  class EventSetup;
}  // namespace edm

namespace ecaldqm {
  struct EcalLSCache {
    std::map<std::string, bool> ByLumiPlotsResetSwitches;
    bool lhcStatusSet_;
  };

  class EcalDQMonitor {
  public:
    EcalDQMonitor(edm::ParameterSet const &);
    virtual ~EcalDQMonitor() noexcept(false);

    static void fillDescriptions(edm::ParameterSetDescription &);

  protected:
    void ecaldqmGetSetupObjects(edm::EventSetup const &);
    void ecaldqmBeginRun(edm::Run const &, edm::EventSetup const &);
    void ecaldqmEndRun(edm::Run const &, edm::EventSetup const &);
    void ecaldqmBeginLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) const;
    void ecaldqmEndLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &);

    template <typename FuncOnWorker>
    void executeOnWorkers_(FuncOnWorker,
                           std::string const &,
                           std::string const & = "",
                           int = 1) const;  // loop over workers and capture exceptions

    std::vector<DQWorker *> workers_;
    std::string const moduleName_;
    const int verbosity_;
  };

  template <typename FuncOnWorker>
  void EcalDQMonitor::executeOnWorkers_(FuncOnWorker _func,
                                        std::string const &_context,
                                        std::string const &_message /* = ""*/,
                                        int _verbThreshold /* = 1*/) const {
    std::for_each(workers_.begin(), workers_.end(), [&](DQWorker *worker) {
      if (verbosity_ > _verbThreshold && !_message.empty())
        edm::LogInfo("EcalDQM") << moduleName_ << ": " << _message << " @ " << worker->getName();
      try {
        _func(worker);
      } catch (std::exception &) {
        edm::LogError("EcalDQM") << moduleName_ << ": Exception in " << _context << " @ " << worker->getName();
        throw;
      }
    });
  }
}  // namespace ecaldqm

#endif
