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

    void initialize();

    void reset();

    virtual void producePlots() = 0;

  protected:
    void source_(unsigned, std::string const&, unsigned, edm::ParameterSet const&);
    float maskQuality_(unsigned, DetId const&, uint32_t, int);
    float maskQuality_(MESet::iterator const&, uint32_t, int);
    float maskPNQuality_(unsigned, EcalPnDiodeDetId const&, int);
    float maskPNQuality_(MESet::iterator const&, int);
    void towerAverage_(unsigned, unsigned, float);

    std::vector<MESet const*> sources_;
  };

}
#endif
