#ifndef RecoECAL_ECALClusters_IslandClusterAlgo_h
#define RecoECAL_ECALClusters_IslandClusterAlgo_h

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/RectangularEtaPhiRegion.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
#include "RecoCaloTools/Navigation/interface/EcalBarrelNavigator.h"
#include "RecoCaloTools/Navigation/interface/EcalEndcapNavigator.h"
#include "Geometry/CaloTopology/interface/EcalBarrelHardcodedTopology.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"

// C/C++ headers
#include <string>
#include <vector>
#include <set>

typedef std::map<DetId, EcalRecHit> RecHitsMap;

class IslandClusterAlgo 
{
 public:
  
  enum EcalPart { barrel = 0, endcap = 1 };
  enum VerbosityLevel { pDEBUG = 0, pWARNING = 1, pINFO = 2, pERROR = 3 }; 

  IslandClusterAlgo() {
  }

  IslandClusterAlgo(double ebst, double ecst, const PositionCalc& posCalc,
                    const std::vector<int>& v_chstatusSeed_Barrel, const std::vector<int>& v_chstatusSeed_Endcap,
                    const std::vector<int>& v_chstatus_Barrel, const std::vector<int>& v_chstatus_Endcap,
                    VerbosityLevel the_verbosity = pERROR) :
    ecalBarrelSeedThreshold(ebst), ecalEndcapSeedThreshold(ecst),
    v_chstatusSeed_Barrel_(v_chstatusSeed_Barrel), v_chstatusSeed_Endcap_(v_chstatusSeed_Endcap),
    v_chstatus_Barrel_(v_chstatus_Barrel), v_chstatus_Endcap_(v_chstatus_Endcap),
    verbosity(the_verbosity) {
    posCalculator_ = posCalc;
  }

  virtual ~IslandClusterAlgo()
    {
    }

  void setVerbosity(VerbosityLevel the_verbosity)
    {
      verbosity = the_verbosity;
    }

  // this is the method that will start the clusterisation
  std::vector<reco::BasicCluster> makeClusters(const EcalRecHitCollection* hits,
                                               const CaloSubdetectorGeometry *geometry,
                                               const CaloSubdetectorTopology *topology_p,
                                               const CaloSubdetectorGeometry *geometryES_p,
                                               EcalPart ecalPart,
					       bool regional = false,
					       const std::vector<RectangularEtaPhiRegion>& regions = std::vector<RectangularEtaPhiRegion>());

  /// point in the space
  typedef math::XYZPoint Point;

 private: 

  //algo to compute position of clusters
  PositionCalc posCalculator_;


  // Energy required for a seed:
  double ecalBarrelSeedThreshold;
  double ecalEndcapSeedThreshold;
  
  // collection of all rechits
  const EcalRecHitCollection *recHits_;

  // The vector of seeds:
  std::vector<EcalRecHit> seeds;

  // The set of used DetID's
  std::set<DetId> used_s;

  // The vector of DetId's in the cluster currently reconstructed
  std::vector< std::pair<DetId, float> > current_v;

  // The vector of clusters
  std::vector<reco::BasicCluster> clusters_v;

  // channels not to be used for seeding
  std::vector<int> v_chstatusSeed_Barrel_;
  std::vector<int> v_chstatusSeed_Endcap_;

  // channels not to be used for clustering
  std::vector<int> v_chstatus_Barrel_;
  std::vector<int> v_chstatus_Endcap_;

  std::vector<int> v_chstatusSeed_;
  std::vector<int> v_chstatus_;

  // The verbosity level
  VerbosityLevel verbosity;

  void mainSearch(const EcalRecHitCollection* hits,
                  const CaloSubdetectorGeometry *geometry_p,
                  const CaloSubdetectorTopology *topology_p,
		  const CaloSubdetectorGeometry *geometryES_p,
                  EcalPart ecalPart);
 
  void searchNorth(const CaloNavigator<DetId> &navigator);
  void searchSouth(const CaloNavigator<DetId> &navigator);
  void searchWest (const CaloNavigator<DetId> &navigator, const CaloSubdetectorTopology* topology);
  void searchEast (const CaloNavigator<DetId> &navigator, const CaloSubdetectorTopology* topology);

  bool shouldBeAdded(EcalRecHitCollection::const_iterator candidate_it, EcalRecHitCollection::const_iterator previous_it);

  void makeCluster(const EcalRecHitCollection* hits,const CaloSubdetectorGeometry *geometry_p,const CaloSubdetectorGeometry *geometryES_p);

 };

#endif
