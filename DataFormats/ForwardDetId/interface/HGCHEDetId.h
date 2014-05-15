#ifndef DataFormats_ForwardDetId_HGCHEDetId_H
#define DataFormats_ForwardDetId_HGCHEDetId_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"


class HGCHEDetId : public DetId {
public:
  enum { Subdet=HGCHE};
  /** Create a null cellid*/
  HGCHEDetId();
  /** Create cellid from raw id (0=invalid tower id) */
  HGCHEDetId(uint32_t rawid);
  /** Constructor from subdetector, zplus, layer, module, cell numbers */
  HGCHEDetId(ForwardSubdetector subdet, int zp, int lay, int mod, 
	     int cellx, int celly);
  /** Constructor from a generic cell id */
  HGCHEDetId(const DetId& id);
  /** Assignment from a generic cell id */
  HGCHEDetId& operator=(const DetId& id);

  /// get the subdetector
  ForwardSubdetector subdet() const { return HGCHE; }
  /// get the z-side of the cell (1/-1)
  int zside() const { return (id_&0x1000000)?(1):(-1); }
  /// get the absolute value of the cell #'s in x and y
  int cellX() const { return (id_>>6)&0x3F; }
  int cellY() const { return id_&0x3F; }
  /// get the module #
  int module() const { return (id_>>12)&0x3F; }
  /// get the layer #
  int layer() const { return (id_>>18)&0x3F; }

  static const HGCHEDetId Undefined;

};

std::ostream& operator<<(std::ostream&,const HGCHEDetId& id);

#endif

