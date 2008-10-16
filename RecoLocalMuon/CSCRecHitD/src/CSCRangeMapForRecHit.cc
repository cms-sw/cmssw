/** CSCRangeMapForRecHit.cc
 *
 *  author: Dominique Fortin -UCR
 */

#include <RecoLocalMuon/CSCRecHitD/src/CSCRangeMapForRecHit.h>

CSCRangeMapForRecHit::CSCRangeMapForRecHit() {}

CSCRangeMapForRecHit::~CSCRangeMapForRecHit() {}

std::pair<CSCDetId,CSCDetIdSameChamberCompare> CSCRangeMapForRecHit::cscChamber(CSCDetId id) {
  
  return std::make_pair(id, CSCDetIdSameChamberCompare());
}

std::pair<CSCDetId,CSCDetIdSameDetLayerCompare> CSCRangeMapForRecHit::cscDetLayer(CSCDetId id) {
  
  return std::make_pair(id, CSCDetIdSameDetLayerCompare());
}

bool CSCDetIdSameChamberCompare::operator()(CSCDetId i1, CSCDetId i2) const {
  if (i1.chamberId() == i2.chamberId())
    return false;

  return (i1<i2);
}

bool CSCDetIdSameDetLayerCompare::operator()(CSCDetId i1, CSCDetId i2) const {
  if ((i1.chamberId() == i2.chamberId()) &&
      (i1.endcap()    == i2.endcap()   ) &&
      (i1.station()   == i2.station()  ) &&
      (i1.ring()      == i2.ring()     ) &&
      (i1.layer()     == i2.layer()    ))
    return false;
    
  return (i1<i2);
}

