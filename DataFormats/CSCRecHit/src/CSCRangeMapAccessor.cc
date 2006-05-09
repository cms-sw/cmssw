/** \file CSCRangeMapAccessor.cc
 *
 *  $Date: 2006/05/02 10:39:53 $
 *  \author Matteo Sani
 */

#include <DataFormats/CSCRecHit/interface/CSCRangeMapAccessor.h>

CSCRangeMapAccessor::CSCRangeMapAccessor() {}

std::pair<CSCDetId,CSCDetIdSameChamberComparator> CSCRangeMapAccessor::cscChamber(CSCDetId id) {
    
    return std::make_pair(id, CSCDetIdSameChamberComparator());
}

bool CSCDetIdSameChamberComparator::operator()(CSCDetId i1, CSCDetId i2) const {
    if (i1.chamber() == i2.chamber() &&
        i1.ring() == i2.ring() &&
        i1.station() == i2.station() &&
        i1.endcap() == i2.endcap()) 
        return false;
    return (i1<i2);
}



