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
  /** Constructor from subdetector, module, signed ieta, signed iphi, depth and type */
  CFCDetId(ForwardSubdetector subdet, int module, int ieta, int iphi, int depth, int type);
  /** Constructor from a generic cell id */
  CFCDetId(const DetId& id);
  /** Assignment from a generic cell id */
  CFCDetId& operator=(const DetId& id);

  /// get the subdetector
  ForwardSubdetector subdet() const { return (ForwardSubdetector)(subdetId()); }
  /// get the z-side of the cell (1/-1)
  int zside() const { return (id_&0x80000)?(1):(-1); }
  /// get the absolute value of the cell ieta
  int ietaAbs() const { return (id_>>11)&0xFF; }
  /// get the cell ieta
  int ieta() const { return zside()*ietaAbs(); }
  /// get the module #
  int module() const { return (id_>>6)&0x1F; }
  /// get the cell iphi
  int signPhi() const { return (id_&0x20)?(1):(-1); }
  int iphi() const { return (id_&0x1F)*signPhi(); }
  /// get the tower depth
  int depth() const { return (id_>>21)&0x7; }
  /// get the fibre type
  int type() const { return (id_>>20)&0x1; }

  static const CFCDetId Undefined;

};

std::ostream& operator<<(std::ostream&,const CFCDetId& id);

#endif
