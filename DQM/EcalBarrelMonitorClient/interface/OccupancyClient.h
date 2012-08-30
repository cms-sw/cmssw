#ifndef OccupancyClient_H
#define OccupancyClient_H

#include "DQWorkerClient.h"

class CaloGeometry;

namespace ecaldqm {

  class OccupancyClient : public DQWorkerClient {
  public:
    OccupancyClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~OccupancyClient() {}

    void beginRun(const edm::Run &, const edm::EventSetup &);

    void producePlots();

    enum MESets {
      kHotDigi,
      kHotRecHitThr,
      kHotTPDigiThr,
      kQualitySummary,
      nMESets
    };

    enum Sources {
      kDigi,
      kRecHitThr,
      kTPDigiThr,
      nSources
    };

    static void setMEOrdering(std::map<std::string, unsigned>&);

  private:
    const CaloGeometry* geometry_;

    int minHits_;
    float deviationThreshold_;
  };

}

#endif

