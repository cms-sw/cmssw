#ifndef DataFormats_ForwardDetId_HGCEEDetId_H
#define DataFormats_ForwardDetId_HGCEEDetId_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"


class HGCEEDetId : public DetId {
public:
  static const int kHGCEECellOffset      = 0;
  static const int kHGCEECellMask        = 0xFFF;
  static const int kHGCEESectorOffset    = 12;
  static const int kHGCEESectorMask      = 0x3F;
  static const int kHGCEESubSectorOffset = 18;
  static const int kHGCEESubSectorMask   = 0x1;
  static const int kHGCEELayerOffset     = 19;
  static const int kHGCEELayerMask       = 0x1F;
  static const int kHGCEEZsideOffset     = 24;
  static const int kHGCEEZsideMask       = 0x1;
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
  
  /** Converter for a geometry cell id */
  HGCEEDetId geometryCell () const {return HGCEEDetId (subdet(), zside(), layer(), sector(), 0, 0);}

  /// get the subdetector
  ForwardSubdetector subdet() const { return HGCEE; }

  /// get the absolute value of the cell #'s in x and y
  int cell() const { return id_&kHGCEECellMask; }

  /// get the sector #
  int sector() const { return (id_>>kHGCEESectorOffset)&kHGCEESectorMask; }

  /// get the degree subsector
  int subsector() const { return ( (id_>>kHGCEESubSectorOffset)&kHGCEESubSectorMask ? 1 : -1); }

  /// get the layer #
  int layer() const { return (id_>>kHGCEELayerOffset)&kHGCEELayerMask; }

  /// get the z-side of the cell (1/-1)
  int zside() const { return ((id_>>kHGCEEZsideOffset) & kHGCEEZsideMask ? 1 : -1); }

  /// consistency check : no bits left => no overhead
  bool isEE()      const { return true; }
  bool isForward() const { return true; }
  
  static const HGCEEDetId Undefined;

};

std::ostream& operator<<(std::ostream&,const HGCEEDetId& id);

#endif
