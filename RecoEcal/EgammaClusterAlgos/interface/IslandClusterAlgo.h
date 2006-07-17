#ifndef RecoECAL_ECALClusters_IslandClusterAlgo_h
#define RecoECAL_ECALClusters_IslandClusterAlgo_h

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
#include "RecoCaloTools/Navigation/interface/EcalBarrelNavigator.h"
#include "RecoCaloTools/Navigation/interface/EcalEndcapNavigator.h"
#include "Geometry/CaloTopology/interface/EcalBarrelHardcodedTopology.h"

// C/C++ headers
#include <string>
#include <vector>
#include <set>

//

enum EcalPart { barrel = 0, endcap = 1 };
typedef std::map<DetId, EcalRecHit> RecHitsMap;


// Less than operator for sorting EcalRecHits according to energy.
class ecalRecHitLess : public std::binary_function<EcalRecHit, EcalRecHit, bool> 
{
 public:
  bool operator()(EcalRecHit x, EcalRecHit y) 
    { 
      return (x.energy() > y.energy()); 
    }
};


class IslandClusterAlgo 
{
 public:
  
  IslandClusterAlgo()
    {
    }

  IslandClusterAlgo(double ebst, double ecst) : 
    ecalBarrelSeedThreshold(ebst), ecalEndcapSeedThreshold(ecst)
    {
    }
  
  virtual ~IslandClusterAlgo()
    {
    }

  // this is the method that will start the clusterisation
  std::vector<reco::BasicCluster> makeClusters(RecHitsMap *the_rechitsMap_p,
					       const CaloSubdetectorGeometry *geometry,
					       CaloSubdetectorTopology *topology_p,
					       EcalPart ecalPart);

  /// point in the space
  typedef math::XYZPoint Point;

 private: 
  
  struct ClusterVars
  {
    double energy;
    double chi2;
    std::vector<DetId> usedHits;
  };

 // Energy required for a seed:
  double ecalBarrelSeedThreshold;
  double ecalEndcapSeedThreshold;
  
  // The map of hits
  RecHitsMap *rechitsMap_p;

  // The vector of seeds:
  std::vector<EcalRecHit> seeds;

  // The set of used DetID's
  std::set<DetId> used_s;

  // The vector of DetId's in the cluster currently reconstructed
  std::vector<DetId> current_v;

  // The vector of clusters
  std::vector<reco::BasicCluster> clusters_v;

  void mainSearch(EcalPart ecalPart, CaloSubdetectorTopology *topology_p); 
 
  void searchNorth(CaloNavigator<DetId> &navigator);

  void searchSouth(CaloNavigator<DetId> &navigator);

  void searchWest (CaloNavigator<DetId> &navigator, CaloSubdetectorTopology &topology);

  void searchEast (CaloNavigator<DetId> &navigator, CaloSubdetectorTopology &topology);

  void makeCluster();

 };

#endif
