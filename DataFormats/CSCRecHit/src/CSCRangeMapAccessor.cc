/** \file CSCRangeMapAccessor.cc
 *
 *  $Date: 2009/02/03 21:56:18 $
 *  \author Matteo Sani
 */

#include "DataFormats/CSCRecHit/interface/CSCRangeMapAccessor.h"
#include <cstdlib>

CSCRangeMapAccessor::CSCRangeMapAccessor() {}

CSCRangeMapAccessor::~CSCRangeMapAccessor() {}

std::pair<CSCDetId,CSCDetIdSameChamberComparator> CSCRangeMapAccessor::cscChamber(CSCDetId id) {
  
  return std::make_pair(id, CSCDetIdSameChamberComparator());
}

std::pair<CSCDetId,CSCDetIdSameDetLayerComparator> CSCRangeMapAccessor::cscDetLayer(CSCDetId id) {
  
  return std::make_pair(id, CSCDetIdSameDetLayerComparator());
}

bool CSCDetIdSameChamberComparator::operator()(CSCDetId i1, CSCDetId i2) const {
  if (i1.chamberId() == i2.chamberId())
    return false;

  return (i1<i2);
}

bool CSCDetIdSameDetLayerComparator::operator()(CSCDetId i1, CSCDetId i2) const {
  bool station = false;
  if (i1.endcap() == i2.endcap() &&
      i1.station() == i2.station())
    station = true;

  // Same DetLayer for station 2,3 and 4
  if ((station) && (i1.station() != 1))
    return false;
  
  // Same DetLayer for station 1
  if ((station) && (i1.station() == 1)) {
  
    int delta = abs(i1.ring() - i2.ring());
    int sum = i1.ring() + i2.ring();
    
    // Same DetLayer: rings 1,4 or rings 2,3
    if ((delta == 0) || (sum == 5))
      return false;
  }

  return (i1<i2);
}







