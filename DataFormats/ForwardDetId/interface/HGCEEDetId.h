#ifndef DataFormats_ForwardDetId_HGCEEDetId_H
#define DataFormats_ForwardDetId_HGCEEDetId_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"


class HGCEEDetId : public DetId {
public:
  enum { Subdet=HGCEE};
  /** Create a null cellid*/
  HGCEEDetId();
  /** Create cellid from raw id (0=invalid tower id) */
  HGCEEDetId(uint32_t rawid);
  /** Constructor from subdetector, zplus, layer, module, cell numbers */
  HGCEEDetId(ForwardSubdetector subdet, int zp, int lay, int mod, int subsec, int cell);
  /** Constructor from a generic cell id */
  HGCEEDetId(const DetId& id);
  /** Assignment from a generic cell id */
  HGCEEDetId& operator=(const DetId& id);

  /// get the subdetector
  ForwardSubdetector subdet() const { return HGCEE; }

  /// get the absolute value of the cell #'s in x and y
  int cell() const { return id_&0xFFFF; }

  /// get the sector #
  int sector() const { return (id_>>16)&0x7F; }

  /// get the degree subsector
  int subsector() const { return ( (id_>>23)&0x1 ? 1 : -1); }

  /// get the layer #
  int layer() const { return (id_>>24)&0x7F; }

  /// get the z-side of the cell (1/-1)
  int zside() const { return ((id_>>31) & 0x1 ? 1 : -1); }

  /// consistency check : no bits left => no overhead
  bool isEE() const { return true; }
  bool isForward() const {  return true; }
  
  static const HGCEEDetId Undefined;

};

std::ostream& operator<<(std::ostream&,const HGCEEDetId& id);

#endif
