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

  /// get the module #
  int module() const { return (id_>>16)&0x1F; }

  /// get the degree subsector
  int subsector() const { return ( (id_>>21)&0x1 ? 1 : -1); }

  /// get the layer #
  int layer() const { return (id_>>22)&0x1F; }

  /// get the z-side of the cell (1/-1)
  int zside() const { return ((id_>>27) & 0x1 ? 1 : -1); }

  /// consistency check
  bool isEE() const { return ((id_>>28) & 0x1); }
  bool isForward() const {  return (((id_>>29)& 0x7)==Forward); }
  
  static const HGCEEDetId Undefined;

};

std::ostream& operator<<(std::ostream&,const HGCEEDetId& id);

#endif
