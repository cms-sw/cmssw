#include "CondFormats/CSCObjects/interface/CSCBadChambers.h"
#include <algorithm>

bool CSCBadChambers::isInBadChamber( int ichamber ) const {

  if ( numberOfChambers() == 0 ) return false;

  std::vector<int>::const_iterator badbegin = chambers.begin();
  std::vector<int>::const_iterator badend   = chambers.end();
  std::vector<int>::const_iterator it = std::find( badbegin, badend, ichamber );
  if ( it != badend ) return true; // ichamber is in the list of bad chambers
  else return false;
}

