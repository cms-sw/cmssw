
#ifndef DataFormats_ForwardDetId_HGCHEDetId_H
#define DataFormats_ForwardDetId_HGCHEDetId_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"


class HGCHEDetId : public DetId {
public:
  /** Create a null cellid*/
  HGCHEDetId();
  /** Create cellid from raw id (0=invalid tower id) */
  HGCHEDetId(uint32_t rawid);
  /** Constructor from subdetector, zplus, layer, module, cell numbers */
  HGCHEDetId(ForwardSubdetector subdet, int zp, int lay, int mod, int subsec, int cell);
  /** Constructor from a generic cell id */
  HGCHEDetId(const DetId& id);
  /** Assignment from a generic cell id */
  HGCHEDetId& operator=(const DetId& id);

  /** Converter for a geometry cell id
   * @param id full EKDetId 
   */
  HGCHEDetId geometryCell () const;

  /// get the subdetector
  ForwardSubdetector subdet() const { return (ForwardSubdetector)(subdetId());}

  /// get the absolute value of the cell #'s in x and y
  int cell() const { return id_&0xFFF; }

  /// get the sector #
  int sector() const { return (id_>>12)&0x3F; }

  /// get the degree subsector
  int subsector() const { return ( (id_>>18)&0x1 ? 1 : -1); }

  /// get the layer #
  int layer() const { return (id_>>19)&0x1F; }

  /// get the z-side of the cell (1/-1)
  int zside() const { return ((id_>>24) & 0x1 ? 1 : -1); }

  /// consistency check
  bool isHE()      const { return true; }
  bool isForward() const { return true; }

  static const HGCHEDetId Undefined;

};

std::ostream& operator<<(std::ostream&,const HGCHEDetId& id);

#endif

