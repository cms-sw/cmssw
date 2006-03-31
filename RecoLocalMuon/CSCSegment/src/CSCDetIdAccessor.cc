#include <RecoLocalMuon/CSCSegment/src/CSCDetIdAccessor.h>

CSCDetIdAccessor::CSCDetIdAccessor() {}

std::pair<CSCDetId,CSCDetIdSameChamberComparator> CSCDetIdAccessor::cscChamber(CSCDetId id) {

    //CSCDetId id(chamber);
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



