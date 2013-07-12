#ifndef DTRecHit_DTEnums_H
#define DTRecHit_DTEnums_H

/** \class DTEnums
 *  Define some useful enums for DTs
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */

#include <ostream>

namespace DTEnums {

  /// Which side of the DT cell
  enum DTCellSide { undefLR = 0, Right = 1, Left = 2 };

}

std::ostream& operator<<(std::ostream& s, DTEnums::DTCellSide p);

#endif

