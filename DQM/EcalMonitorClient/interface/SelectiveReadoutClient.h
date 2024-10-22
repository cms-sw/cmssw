#ifndef SelectiveReadoutClient_H
#define SelectiveReadoutClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class SelectiveReadoutClient : public DQWorkerClient {
  public:
    SelectiveReadoutClient();
    ~SelectiveReadoutClient() override {}

    void producePlots(ProcessType) override;
  };

}  // namespace ecaldqm

#endif
