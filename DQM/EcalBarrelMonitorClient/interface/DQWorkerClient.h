#ifndef DQWorkerClient_H
#define DQWorkerClient_H

#include <utility>

#include "DQM/EcalCommon/interface/DQWorker.h"

namespace ecaldqm {
  class DQWorkerClient : public DQWorker {
  public:
    DQWorkerClient(edm::ParameterSet const&, edm::ParameterSet const&, std::string const&);
    virtual ~DQWorkerClient() {}

    void endLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &);

    void bookSummaries();

    void reset();
    void initialize();

    virtual void producePlots() = 0;

    enum Quality {
      kBad = 0,
      kGood = 1,
      kUnknown = 2,
      kMBad = 3,
      kMGood = 4,
      kMUnknown = 5
    };

  protected:
    void towerAverage_(MESet*, MESet const*, float);
    bool using_(std::string const& _s) { return usedSources_.find(_s) != usedSources_.end(); }

    ConstMESetCollection sources_;
    std::set<std::string> usedSources_;
    std::set<std::string> qualitySummaries_;
  };

}
#endif
