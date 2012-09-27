#ifndef OccupancyClient_H
#define OccupancyClient_H

#include "DQM/EcalCommon/interface/DQWorkerClient.h"

class CaloGeometry;

namespace ecaldqm {

  class OccupancyClient : public DQWorkerClient {
  public:
    OccupancyClient(const edm::ParameterSet &, const edm::ParameterSet &);
    ~OccupancyClient() {}

    void beginRun(const edm::Run &, const edm::EventSetup &);

    void bookMEs();

    void producePlots();

    enum MESets {
      kHotDigi,
      kHotRecHitThr,
      kHotTPDigiThr,
      kQualitySummary,
      nMESets
    };

    static void setMEData(std::vector<MEData>&);

    enum Sources {
      sDigi,
      sRecHitThr,
      sTPDigiThr,
      nSources
    };

  private:
    const CaloGeometry* geometry_;

    int minHits_;
    float deviationThreshold_;
  };

}

#endif

