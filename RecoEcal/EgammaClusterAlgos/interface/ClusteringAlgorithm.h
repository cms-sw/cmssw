#ifndef RecoECAL_ECALClusters_ClusteringAlgorithm_h
#define RecoECAL_ECALClusters_ClusteringAlgorithm_h


#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "RecoEcal/EgammaClusterAlgos/interface/PositionAwareHit.h"


// C/C++ headers
#include <string>
#include <vector>

//

const double defaultSeedThreshold = 2;

class ClusteringAlgorithm 
{
 public:
  
  ClusteringAlgorithm()
    {
      ecalBarrelSeedThreshold = defaultSeedThreshold;
    }

  ClusteringAlgorithm(double ebst, double ecst) : ecalBarrelSeedThreshold(ebst)
    {
    }
  
  virtual ~ClusteringAlgorithm()
    {
    }

  // this is the method that will start the clusterisation
  std::vector<reco::BasicCluster> makeClusters(EcalRecHitCollection & rechits,
					       const CaloSubdetectorGeometry *geometry);
  /// point in the space
    typedef math::XYZPoint Point;

 protected: 
  
  // Energy required for a seed:
  double ecalBarrelSeedThreshold;
  
  // Map of RecHits
  std::map<EBDetId, PositionAwareHit> rechits_m;

  // The vector of seeds:
  std::vector<PositionAwareHit> seeds;

  // The vector of clusters
  std::vector<reco::BasicCluster> clusters;

  virtual void mainSearch(const CaloSubdetectorGeometry *geometry) = 0; //the real clustering algorithm
  Point getECALposition(std::vector<reco::EcalRecHitData> recHits, const CaloSubdetectorGeometry *geometry);//Position determination

};

#endif
