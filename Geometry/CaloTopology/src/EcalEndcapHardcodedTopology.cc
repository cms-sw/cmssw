#include "Geometry/CaloTopology/interface/EcalEndcapHardcodedTopology.h"

EEDetId EcalEndcapHardcodedTopology::incrementIx(const EEDetId& id) const {
  if (! (EEDetId::validDetId(id.ix()+1,id.iy(),id.zside()) ) ) return EEDetId(0); // null det id
  else return EEDetId(id.ix()+1,id.iy(),id.zside());	 
}

EEDetId EcalEndcapHardcodedTopology::decrementIx(const EEDetId& id) const {
  if (! (EEDetId::validDetId(id.ix()-1,id.iy(),id.zside()) ) ) return EEDetId(0); // null det id
  else return EEDetId(id.ix()-1,id.iy(),id.zside());	 
}

EEDetId EcalEndcapHardcodedTopology::incrementIy(const EEDetId& id) const {
  if (! (EEDetId::validDetId(id.ix(),id.iy()+1,id.zside()) ) ) return EEDetId(0); // null det id
  else return EEDetId(id.ix(),id.iy()+1,id.zside());	 
}

EEDetId EcalEndcapHardcodedTopology::decrementIy(const EEDetId& id) const {
  if (! (EEDetId::validDetId(id.ix(),id.iy()-1,id.zside()) ) ) return EEDetId(0); // null det id
  else return EEDetId(id.ix(),id.iy()-1,id.zside());	 
}
