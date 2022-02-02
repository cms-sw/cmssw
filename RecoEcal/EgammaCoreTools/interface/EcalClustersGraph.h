#ifndef RecoEcal_EgammaCoreTools_EcalClustersGraph_h
#define RecoEcal_EgammaCoreTools_EcalClustersGraph_h

/**
   \file
   Tools for manipulating ECAL Clusters as graphs
   \author Davide Valsecchi, Badder Marzocchi
   \date 05 October 2020
*/

#include <vector>
#include <array>
#include <fstream>
#include <algorithm>
#include "TRandom.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"

#include "RecoEcal/EgammaCoreTools/interface/CalibratedPFCluster.h"
#include "RecoEcal/EgammaCoreTools/interface/GraphMap.h"
#include "RecoEcal/EgammaCoreTools/interface/DeepSCGraphEvaluation.h"

using namespace std;
using namespace reco;
namespace reco {

  class EcalClustersGraph {
    typedef std::shared_ptr<CalibratedPFCluster> CalibratedClusterPtr;
    typedef std::vector<CalibratedClusterPtr> CalibratedClusterPtrVector;

  private:
    CalibratedClusterPtrVector clusters_;
    uint nSeeds_;
    uint nCls_;

    //To compute the input variables
    const CaloTopology* topology_;
    const CaloSubdetectorGeometry* ebGeom_;
    const CaloSubdetectorGeometry* eeGeom_;
    const EcalRecHitCollection* recHitsEB_;
    const EcalRecHitCollection* recHitsEE_;
    const SCProducerCache* SCProducerCache_;

    // GraphMap for handling all the windows and scores
    const static inline std::vector<uint> NODES_CATEGORIES = {0, 1};  // 0 =normal cluster, 1 seed
    GraphMap graphMap_;
    std::vector<std::pair<uint, std::vector<uint>>> finalSuperClusters_;
    std::array<float, 3> locCov_;
    std::pair<double, double> widths_;
    float threshold_;
    DeepSCInputs inputs_;

    std::ofstream outfile;
    
  public:
    EcalClustersGraph(CalibratedClusterPtrVector clusters,
                      int nSeeds,
                      const CaloTopology* topology,
                      const CaloSubdetectorGeometry* ebGeom,
                      const CaloSubdetectorGeometry* eeGeom,
                      const EcalRecHitCollection* recHitsEB,
                      const EcalRecHitCollection* recHitsEE,
                      const SCProducerCache* cache);

    std::vector<int> clusterPosition(const CaloCluster* cluster);

    double deltaPhi(double seed_phi, double cluster_phi) {
      double dphi = seed_phi - cluster_phi;
      if (dphi > TMath::Pi())
        dphi -= 2 * TMath::Pi();
      if (dphi < -TMath::Pi())
        dphi += 2 * TMath::Pi();
      return dphi;
    }

    double deltaEta(double seed_eta, double cluster_eta) {
      double deta = 0.;
      if (seed_eta > 0.)
        deta = cluster_eta - seed_eta;
      if (seed_eta <= 0.)
        deta = seed_eta - cluster_eta;
      return deta;
    }
    std::vector<double> dynamicWindow(double seedEta);

    std::pair<double, double> computeCovariances(const CaloCluster* cluster);
    std::vector<double> computeShowerShapes(const CaloCluster* cluster, bool full5x5);
    std::vector<double> computeVariables(const CaloCluster* seed, const CaloCluster* cluster);
    std::vector<std::vector<double>> fillHits(const CaloCluster* cluster);
    std::vector<double> computeWindowVariables(const std::vector<std::vector<double>>& clusters);

    void fillVariables();

    double scoreThreshold(const CaloCluster* cluster);
    void initWindows();
    void clearWindows();

    void setThresholds();
    void evaluateScores();
    void selectClusters();

    void printDebugInfo();
    std::vector<std::pair<CalibratedClusterPtr, CalibratedClusterPtrVector>> getWindows();
  };

}  // namespace reco
#endif
