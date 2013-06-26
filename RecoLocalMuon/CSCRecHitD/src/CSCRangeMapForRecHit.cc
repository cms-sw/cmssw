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
  if (i1.chamberId() == i2.chamberId()) return false; // this removes layer bit and then uses DetId::op==
  return (i1<i2);
}

bool CSCDetIdSameDetLayerCompare::operator()(CSCDetId i1, CSCDetId i2) const {
  if (i1 == i2 ) return false; // use DetId::op==
  return (i1<i2);
}

