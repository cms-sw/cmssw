#ifndef SelectiveReadoutClient_H
#define SelectiveReadoutClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class SelectiveReadoutClient : public DQWorkerClient {
  public:
    SelectiveReadoutClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~SelectiveReadoutClient() {}

    void producePlots();

    enum MESets {
      kFRDropped,
      kZSReadout,
      kFR,
      kRUForced,
      kZS1,
      kHighInterest,
      kMedInterest,
      kLowInterest,
      nMESets
    };

    enum Sources {
      kFlagCounterMap,
      kRUForcedMap,
      kFullReadoutMap,
      kZS1Map,
      kZSMap,
      kZSFullReadoutMap,
      kFRDroppedMap,
      kHighIntMap,
      kMedIntMap,
      kLowIntMap,
      nSources
    };

    static void setMEOrdering(std::map<std::string, unsigned>&);

  };

}

#endif
