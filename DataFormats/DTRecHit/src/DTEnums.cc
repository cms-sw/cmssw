/** \file
 *
 *  $Date: 2006/06/29 17:18:27 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DataFormats/DTRecHit/interface/DTEnums.h"

using namespace std;

ostream& operator<<( ostream& s, DTEnums::DTCellSide p) {
  switch(p) {
    case DTEnums::undefLR:
      return s << "undefined";
    case DTEnums::Right:
      return s << "Right";
    case DTEnums::Left:
      return s << "Left";
  }
  return s;
}
