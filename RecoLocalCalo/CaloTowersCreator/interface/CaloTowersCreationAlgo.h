#ifndef RECOLOCALCALO_CALOTOWERSCREATOR_CALOTOWERSCREATIONALGO_H
#define RECOLOCALCALO_CALOTOWERSCREATOR_CALOTOWERSCREATIONALGO_H 1

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

/** \class CaloTowersCreationAlgo
  *  
  * $Date: $
  * $Revision: $
  * \author J. Mans - Minnesota
  */
class CaloTowersCreationAlgo {
public:
  CaloTowersCreationAlgo(const HcalTopology* topo, const CaloGeometry* geo);

  bool create(const CaloTowerDetId& id, CaloTowerCollection& destCollection,
	      const HBHERecHitCollection& hbhe, 
	      const HORecHitCollection& ho, 
	      const HFRecHitCollection& hf) const; // eventually will need ECAL also.

private:
  void hadForTower(const CaloTowerDetId& id, const HBHERecHitCollection& hbhe, std::vector<const HBHERecHit*>& hits) const;
  const HORecHit* outerForTower(const CaloTowerDetId& id, const HORecHitCollection& ho) const;
  void forwardForTower(const CaloTowerDetId& id, const HFRecHitCollection& hf, std::vector<const HFRecHit*>& hits) const;
  
  void reconstructTower(CaloTower& tower,const std::vector<const HBHERecHit*>& hbhe, const HORecHit* ho, const std::vector<const HFRecHit*>& hf) const;
  const HcalTopology* theTopology;
  const CaloGeometry* theGeometry;
};

#endif
