#ifndef RECOLOCALCALO_HFCLUSTERPRODUCER_HFCLUSTERALGO_H
#define RECOLOCALCALO_HFCLUSTERPRODUCER_HFCLUSTERALGO_H 1

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/EgammaReco/interface/HFEMClusterShape.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/HFEMClusterShapeAssociation.h"
#include "DataFormats/EgammaReco/interface/HFEMClusterShape.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include <map>
#include <list>
using namespace reco;
/** \class HFClusterAlgo
 
  * \author K. Klapoetke -- Minnesota
  */
class HFClusterAlgo {
public:
  HFClusterAlgo(); 

    void setup();

  /** Analyze the hits */
  void clusterize(const HFRecHitCollection& hf, 
		  const CaloGeometry& geom,
		  reco::HFEMClusterShapeCollection& clusters,
		  reco::BasicClusterCollection& BasicClusters,
		  reco::SuperClusterCollection& SuperClusters);


private:
  friend class CompareHFCompleteHitET;
  friend class CompareHFCore;
  struct HFCompleteHit {
    HcalDetId id;
    double energy, et;
  };

  void makeCluster(const HcalDetId& seedid,
		   const HFRecHitCollection& hf, 
		   const CaloGeometry& geom,
		   HFEMClusterShape& clusShp,
		   BasicCluster& Bclus,
		   SuperCluster& SClus);
};

#endif 
