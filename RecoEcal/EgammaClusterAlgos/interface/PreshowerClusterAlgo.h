#ifndef RecoEcal_EgammaClusterAlgos_PreshowerClusterAlgo_h
#define RecoEcal_EgammaClusterAlgos_PreshowerClusterAlgo_h
//
// $Id: PreshowerClusterAlgo.h,v 1.2 2006/06/11 18:02:54 rahatlou Exp $
//

#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "RecoCaloTools/Navigation/interface/EcalPreshowerNavigator.h"
//#include "RecoEcal/EgammaClusterAlgos/interface/LogPositionCalc.h"

// C/C++ headers
#include <string>
#include <vector>
#include <set>
//#include <fstream>

typedef std::map<DetId, EcalRecHit> RecHitsMap;

enum ESplane { plane1 = 1, plane2 = 2 };

#ifndef DebugLevelEnum
#define DebugLevelEnum
enum DebugLevel{ pDEBUG = 0, pINFO = 1, pHISTO = 2, pERROR = 3 }; 
#endif


class PreshowerClusterAlgo {

 public:

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

