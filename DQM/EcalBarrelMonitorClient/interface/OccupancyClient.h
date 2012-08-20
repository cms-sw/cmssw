#ifndef OccupancyClient_H
#define OccupancyClient_H

#include "DQWorkerClient.h"

class CaloGeometry;

namespace ecaldqm {

  class OccupancyClient : public DQWorkerClient {
  public:
    OccupancyClient(const edm::ParameterSet &);
    ~OccupancyClient() {}

    void beginRun(const edm::Run &, const edm::EventSetup &);

    void bookMEs();

    void producePlots();

    enum MESets {
      kHotDigi,
      kHotRecHitThr,
      kHotTPDigiThr,
      kQualitySummary,
      nTargets,
      sDigi = 0,
      sRecHitThr,
      sTPDigiThr,
      nSources,
      nMESets = nTargets + nSources
    };

    static void setMEData(std::vector<MEData>&);

  private:
    const CaloGeometry* geometry_;

    int minHits_;
    float deviationThreshold_;
  };

}

#endif

