#ifndef DQWorkerClient_H
#define DQWorkerClient_H

#include <utility>

#include "DQM/EcalCommon/interface/DQWorker.h"

#include "CondFormats/EcalObjects/interface/EcalDQMChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDQMTowerStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

namespace ecaldqm {
  class DQWorkerClient : public DQWorker {
  public:
    DQWorkerClient(const edm::ParameterSet&, const edm::ParameterSet&, std::string const&);
    virtual ~DQWorkerClient() {}

    virtual void endLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &);

    virtual void initialize();

    void reset();

    virtual void producePlots() = 0;

    enum Sources {
      nSources
    };

    static EcalDQMChannelStatus const* channelStatus;
    static EcalDQMTowerStatus const* towerStatus;

  protected:
    void source_(unsigned, std::string const&, unsigned, edm::ParameterSet const&);
    void fillQuality_(unsigned, DetId const&, uint32_t, float);

    std::vector<MESet const*> sources_;
  };

}
#endif
