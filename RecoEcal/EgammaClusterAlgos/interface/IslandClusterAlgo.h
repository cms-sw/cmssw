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
  
  enum EcalPart { barrel = 0, endcap = 1 };
  enum VerbosityLevel { pDEBUG = 0, pWARNING = 1, pINFO = 2, pERROR = 3 }; 

  IslandClusterAlgo()
    {
    }

  IslandClusterAlgo(double ebst, double ecst, VerbosityLevel the_verbosity = pERROR) : 
    ecalBarrelSeedThreshold(ebst), ecalEndcapSeedThreshold(ecst), verbosity(the_verbosity)
    {
    }
  
  virtual ~IslandClusterAlgo()
    {
    }

  void setVerbosity(VerbosityLevel the_verbosity)
    {
      verbosity = the_verbosity;
    }

  // this is the method that will start the clusterisation
  std::vector<reco::BasicCluster> makeClusters(RecHitsMap *the_rechitsMap_p,
					       const CaloSubdetectorGeometry *geometry,
					       const CaloSubdetectorTopology *topology_p,
					       EcalPart ecalPart);

  /// point in the space
  typedef math::XYZPoint Point;

 private: 
  
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

  // The verbosity level
  VerbosityLevel verbosity;

  void mainSearch(EcalPart ecalPart, const CaloSubdetectorTopology *topology_p); 
 
  void searchNorth(const CaloNavigator<DetId> &navigator);
  void searchSouth(const CaloNavigator<DetId> &navigator);
  void searchWest (const CaloNavigator<DetId> &navigator, const CaloSubdetectorTopology* topology);
  void searchEast (const CaloNavigator<DetId> &navigator, const CaloSubdetectorTopology* topology);

  bool shouldBeAdded(RecHitsMap::iterator candidate_it, RecHitsMap::iterator previous_it);

  void makeCluster();

 };

#endif
