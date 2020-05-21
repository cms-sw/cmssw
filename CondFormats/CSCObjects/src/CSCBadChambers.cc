#include "CondFormats/CSCObjects/interface/CSCBadChambers.h"
#include <algorithm>

bool CSCBadChambers::isInBadChamber(IndexType ichamber) const {
  if (numberOfChambers() == 0)
    return false;

  auto badbegin = chambers.begin();
  auto badend = chambers.end();
  auto it = std::find(badbegin, badend, ichamber);
  if (it != badend)
    return true;  // ichamber is in the list of bad chambers
  else
    return false;
}

bool CSCBadChambers::isInBadChamber(const CSCDetId& id) const {
  if (numberOfChambers() == 0)
    return false;

  return isInBadChamber(chamberIndex(id.endcap(), id.station(), id.ring(), id.chamber()));
}
