#ifndef SelectiveReadoutClient_H
#define SelectiveReadoutClient_H

#include "DQM/EcalCommon/interface/DQWorkerClient.h"

namespace ecaldqm {

  class SelectiveReadoutClient : public DQWorkerClient {
  public:
    SelectiveReadoutClient(const edm::ParameterSet &, const edm::ParameterSet &);
    ~SelectiveReadoutClient() {}

    void producePlots();

    enum MESets {
      kFRDropped,
      kZSReadout,
      kFR,
      kRUForced,
      kZS1,
      nMESets
    };

    static void setMEData(std::vector<MEData>&);

    enum Sources {
      sFlagCounterMap, // h2f counter
      sRUForcedMap, // h2f counter
      sFullReadoutMap, // h2f counter
      sZS1Map, // h2f counter
      sZSMap, // h2f counter
      sZSFullReadoutMap, // h2f counter
      sFRDroppedMap, // h2f counter
      nSources
    };
  };

}

#endif
