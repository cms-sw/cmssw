#ifndef RecoEcal_EgammaClusterAlgos_PreshowerClusterAlgo_h
#define RecoEcal_EgammaClusterAlgos_PreshowerClusterAlgo_h
//
// $Id: PreshowerClusterAlgo.h,v 1.4 2006/07/05 18:13:19 dbanduri Exp $
//
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "RecoCaloTools/Navigation/interface/EcalPreshowerNavigator.h"

// C/C++ headers
#include <string>
#include <vector>
#include <set>

class CaloSubdetectorGeometry;
class CaloSubdetectorTopology;


class PreshowerClusterAlgo {

 public:
   enum ESplane { plane1 = 1, plane2 = 2 };
   enum DebugLevel{ pDEBUG = 0, pINFO = 1, pHISTO = 2, pERROR = 3 }; 

   typedef std::map<DetId, EcalRecHit> RecHitsMap;
   typedef math::XYZPoint Point;

   PreshowerClusterAlgo() : 
   preshStripEnergyCut_(0.), preshClusterEnergyCut_(0.), preshSeededNstr_(15), debugLevel_(pINFO)
   {}

   PreshowerClusterAlgo(double stripEnergyCut, double clusterEnergyCut, int nStripCut, DebugLevel debugLevel = pINFO) :
   preshStripEnergyCut_(stripEnergyCut), preshClusterEnergyCut_(clusterEnergyCut), preshSeededNstr_(nStripCut), debugLevel_(debugLevel)
   {}

   ~PreshowerClusterAlgo() {};

   reco::PreshowerCluster makeOneCluster(ESDetId strip, 
                                         RecHitsMap *rechits_map,
				         reco::BasicClusterRefVector::iterator basicClust_ref,
                                         const CaloSubdetectorGeometry*& geometry_p,
					 CaloSubdetectorTopology*& topology_p);

   bool goodStrip(RecHitsMap::iterator candidate_it, ESDetId ID);

   void findRoad(ESDetId strip, EcalPreshowerNavigator theESNav, int plane);

 private:
  
   double preshStripEnergyCut_;
   double preshClusterEnergyCut_;
   int preshSeededNstr_;
   int debugLevel_;

   std::vector<ESDetId> road_2d;

   // The map of hits
   RecHitsMap *rechits_map;
   // The set of used DetID's
   std::set<DetId> used_s;

};
#endif

