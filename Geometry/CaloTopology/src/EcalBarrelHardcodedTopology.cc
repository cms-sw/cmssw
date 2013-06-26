#include "Geometry/CaloTopology/interface/EcalBarrelHardcodedTopology.h"

EBDetId EcalBarrelHardcodedTopology::incrementIeta(const EBDetId& id) const {
  if (id.ieta()==EBDetId::MAX_IETA) return EBDetId(0); // null det id
  else if (id.ieta()==-1) return EBDetId(1,id.iphi());
  else return EBDetId(id.ieta()+1,id.iphi());
}

EBDetId EcalBarrelHardcodedTopology::decrementIeta(const EBDetId& id) const {
  if (id.ieta()==-EBDetId::MAX_IETA) return EBDetId(0); // null det id
  else if (id.ieta()==1) return EBDetId(-1,id.iphi());
  else return EBDetId(id.ieta()-1,id.iphi());
}

EBDetId EcalBarrelHardcodedTopology::incrementIphi(const EBDetId& id) const {
  if (id.iphi()==EBDetId::MAX_IPHI) return EBDetId(id.ieta(),EBDetId::MIN_IPHI);
  else return EBDetId(id.ieta(),id.iphi()+1);
}

EBDetId EcalBarrelHardcodedTopology::decrementIphi(const EBDetId& id) const {
  if (id.iphi()==EBDetId::MIN_IPHI) return EBDetId(id.ieta(),EBDetId::MAX_IPHI);
  else return EBDetId(id.ieta(),id.iphi()-1);
}
