#ifndef RecoEcal_EgammaClusterAlgos_PreshowerClusterAlgo_h
#define RecoEcal_EgammaClusterAlgos_PreshowerClusterAlgo_h
//
// $Id: PreshowerClusterAlgo.h,v 1.9 2006/07/20 18:51:59 rahatlou Exp $
//

//#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "RecoCaloTools/Navigation/interface/EcalPreshowerNavigator.h"

// C/C++ headers
#include <string>
#include <vector>
#include <set>

class CaloSubdetectorGeometry;
class CaloSubdetectorTopology;


class PreshowerClusterAlgo {


 public:

   enum DebugLevel { pDEBUG = 0, pINFO = 1, pERROR = 2 }; 

   typedef math::XYZPoint Point;

   typedef std::map<DetId, EcalRecHit> RecHitsMap;
   typedef std::set<DetId> HitsID;

   PreshowerClusterAlgo() : 
   preshStripEnergyCut_(0.), preshClusterEnergyCut_(0.), preshSeededNstr_(15), debugLevel_(pINFO)
   {}

   PreshowerClusterAlgo(double stripEnergyCut, double clusterEnergyCut, int nStripCut, DebugLevel debugLevel = pINFO) :
   preshStripEnergyCut_(stripEnergyCut), preshClusterEnergyCut_(clusterEnergyCut), preshSeededNstr_(nStripCut), debugLevel_(debugLevel)
   {}

   ~PreshowerClusterAlgo() {};

   reco::PreshowerCluster makeOneCluster(ESDetId strip,
					 HitsID *used_strips,
                                         RecHitsMap *rechits_map,					 
                                         const CaloSubdetectorGeometry*& geometry_p,
                                         CaloSubdetectorTopology*& topology_p);

   bool goodStrip(RecHitsMap::iterator candidate_it);

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
   HitsID *used_s;

};
#endif

