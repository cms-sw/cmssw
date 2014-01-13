#ifndef DataFormats_ForwardDetId_CFCDetId_H
#define DataFormats_ForwardDetId_CFCDetId_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"


class CFCDetId : public DetId {
public:
  /** Create a null cellid*/
  CFCDetId();
  /** Create cellid from raw id (0=invalid tower id) */
  CFCDetId(uint32_t rawid);
  /** Constructor from subdetector, signed ieta,iphi and depth */
  CFCDetId(ForwardSubdetector subdet, int ieta, int iphi, int depth);
  /** Constructor from a generic cell id */
  CFCDetId(const DetId& id);
  /** Assignment from a generic cell id */
  CFCDetId& operator=(const DetId& id);

  /// get the subdetector
  ForwardSubdetector subdet() const { return (ForwardSubdetector)(subdetId()); }
  /// get the z-side of the cell (1/-1)
  int zside() const { return (id_&0x1000000)?(1):(-1); }
  /// get the absolute value of the cell ieta
  int ietaAbs() const { return (id_>>10)&0x3FF; }
  /// get the cell ieta
  int ieta() const { return zside()*ietaAbs(); }
  /// get the cell iphi
  int iphi() const { return id_&0x3FF; }
  /// get the tower depth
  int depth() const { return (id_>>20)&0xF; }

  static const CFCDetId Undefined;

};

std::ostream& operator<<(std::ostream&,const CFCDetId& id);

#endif
