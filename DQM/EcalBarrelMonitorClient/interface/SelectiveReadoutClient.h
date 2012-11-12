#ifndef SelectiveReadoutClient_H
#define SelectiveReadoutClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class SelectiveReadoutClient : public DQWorkerClient {
  public:
    SelectiveReadoutClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~SelectiveReadoutClient() {}

    void producePlots();

  };

}

#endif
