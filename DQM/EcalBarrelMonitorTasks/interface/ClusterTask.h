#ifndef ClusterTask_H
#define ClusterTask_H

#include "DQM/EcalCommon/interface/DQWorkerTask.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

class CaloTopology;
class CaloSubdetectorGeometry;

namespace ecaldqm {

  class ClusterTask : public DQWorkerTask {
  public:
    ClusterTask(const edm::ParameterSet &, const edm::ParameterSet&);
    ~ClusterTask();

    void bookMEs();

    bool filterRunType(const std::vector<short>&);

    void beginRun(const edm::Run &, const edm::EventSetup &);
    void beginEvent(const edm::Event &, const edm::EventSetup &);

    void analyze(const void*, Collections);

    void runOnRecHits(const EcalRecHitCollection &, Collections);
    void runOnBasicClusters(const reco::BasicClusterCollection &, Collections);
    void runOnSuperClusters(const reco::SuperClusterCollection &, Collections);

    enum MESets {
      kBCEMap, // profile2d
      kBCEMapProjEta, // profile
      kBCEMapProjPhi, // profile
      kBCOccupancy, // h2f
      kBCOccupancyProjEta, // h1f
      kBCOccupancyProjPhi, // h1f
      kBCSizeMap, // profile2d
      kBCSizeMapProjEta, // profile
      kBCSizeMapProjPhi, // profile
      kBCE, // h1f
      kBCNum, // h1f for EB & EE
      kBCSize, // h1f for EB & EE
      kSCE, // h1f
      kSCELow, // h1f
      kSCSeedEnergy, // h1f
      kSCClusterVsSeed, // h2f
      kSCSeedOccupancy, // h2f
      kSingleCrystalCluster, // h2f
      kSCNum, // h1f
      kSCNBCs, // h1f
      kSCNcrystals, // h1f
      kSCR9, // h1f
      kPi0, // h1f
      kJPsi, // h1f
      kZ, // h1f
      kHighMass, // h1f
      nMESets
    };

    // needs to be declared in each derived class
    static void setMEData(std::vector<MEData>&);

  private:
    const CaloTopology *topology_;
    const CaloSubdetectorGeometry* ebGeometry_;
    const CaloSubdetectorGeometry* eeGeometry_;
    const EcalRecHitCollection *ebHits_, *eeHits_;
    int ievt_;
    float lowEMax_;
    int massCalcPrescale_;
  };

  inline void ClusterTask::analyze(const void* _p, Collections _collection){
    switch(_collection){
    case kEBRecHit:
    case kEERecHit:
      runOnRecHits(*static_cast<const EcalRecHitCollection*>(_p), _collection);
      break;
    case kEBBasicCluster:
    case kEEBasicCluster:
      runOnBasicClusters(*static_cast<const reco::BasicClusterCollection*>(_p), _collection);
      break;
    case kEBSuperCluster:
    case kEESuperCluster:
      runOnSuperClusters(*static_cast<const reco::SuperClusterCollection*>(_p), _collection);
      break;
    default:
      break;
    }
  }

}

#endif

