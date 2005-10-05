#include "RecoLocalCalo/CaloTowersCreator/interface/CaloTowersCreationAlgo.h"

/* 
   Algorithm is to loop over all possible towers and see if any of the tower constituents are present.
*/
void CaloTowersCreationAlgo::hadForTower(const CaloTowerDetId& id, const HBHERecHitCollection& hbhe, std::vector<const HBHERecHit*>& hits) const {
  // create all valid HcalBarrel and HcalEndcap detids at this position
  HBHERecHitCollection::const_iterator i;

  if (id.ietaAbs()<=theTopology->lastHBRing()) {
    i=hbhe.find(HcalDetId(HcalBarrel,id.ieta(),id.iphi(),1));
    if (i!=hbhe.end()) hits.push_back(&(*i));
    i=hbhe.find(HcalDetId(HcalBarrel,id.ieta(),id.iphi(),2));
    if (i!=hbhe.end()) hits.push_back(&(*i));
  }
  if (id.ietaAbs()>=theTopology->firstHERing() && id.ietaAbs()<=theTopology->lastHERing()) {
    i=hbhe.find(HcalDetId(HcalEndcap,id.ieta(),id.iphi(),1));
    if (i!=hbhe.end()) hits.push_back(&(*i));
    i=hbhe.find(HcalDetId(HcalEndcap,id.ieta(),id.iphi(),2));
    if (i!=hbhe.end()) hits.push_back(&(*i));
    i=hbhe.find(HcalDetId(HcalEndcap,id.ieta(),id.iphi(),3));
    if (i!=hbhe.end()) hits.push_back(&(*i));
  }
}

const HORecHit* CaloTowersCreationAlgo::outerForTower(const CaloTowerDetId& id, const HORecHitCollection& ho) const {
  // create any HcalOuter detid at this position
  HORecHitCollection::const_iterator i;
  const HORecHit* retval=0;

  if (id.ietaAbs()<=theTopology->lastHORing()) {
    i=ho.find(HcalDetId(HcalOuter,id.ieta(),id.iphi(),4)); // TODO: replace magic number!
    if (i!=ho.end()) retval=&(*i);
  }
  return retval;
}

void CaloTowersCreationAlgo::forwardForTower(const CaloTowerDetId& id, const HFRecHitCollection& hf, std::vector<const HFRecHit*>& hits) const {
  // create all valid HcalForward detids at this position
  HFRecHitCollection::const_iterator i;

  if (id.ietaAbs()>=theTopology->firstHFRing()) {
    i=hf.find(HcalDetId(HcalForward,id.ieta(),id.iphi(),1)); 
    if (i!=hf.end()) hits.push_back(&(*i));
    i=hf.find(HcalDetId(HcalForward,id.ieta(),id.iphi(),2)); 
    if (i!=hf.end()) hits.push_back(&(*i));
  }
}

void CaloTowersCreationAlgo::reconstructTower(CaloTower& tower,const std::vector<const HBHERecHit*>& hbhe, const HORecHit* ho, const std::vector<const HFRecHit*>& hf) const {
  
  

}

bool CaloTowersCreationAlgo::create(const CaloTowerDetId& id, CaloTowerCollection& destCollection,
				    const HBHERecHitCollection& hbhe, 
				    const HORecHitCollection& ho, 
				    const HFRecHitCollection& hf) const {
  std::vector<const HBHERecHit*> hbhe_hits;
  hadForTower(id,hbhe,hbhe_hits);
  const HORecHit*  ho_hit = outerForTower(id,ho);
  std::vector<const HFRecHit*> hf_hits;
  forwardForTower(id,hf,hf_hits);
  
  bool anyhits=(ho_hit!=0 || !hbhe_hits.empty() || !hf_hits.empty());
  if (anyhits) {
    CaloTower tower(id);
    // get the tower's location from the geometry

    // set the tower's values 
    reconstructTower(tower,hbhe_hits,ho_hit, hf_hits);
    destCollection.push_back(tower);
  }

  return anyhits;
}
