#ifndef DataFormats_ForwardDetId_HGCalDetId_H
#define DataFormats_ForwardDetId_HGCalDetId_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"


class HGCalDetId : public DetId {

public:
  static const int kHGCalCellOffset      = 0;
  static const int kHGCalCellMask        = 0xFFFF;
  static const int kHGCalSectorOffset    = 16;
  static const int kHGCalSectorMask      = 0x7F;
  static const int kHGCalSubSectorOffset = 23;
  static const int kHGCalSubSectorMask   = 0x1;
  static const int kHGCalLayerOffset     = 24;
  static const int kHGCalLayerMask       = 0x7F;
  static const int kHGCalZsideOffset     = 31;
  static const int kHGCalZsideMask       = 0x1;

  /** Create a null cellid*/
  HGCalDetId();
  /** Create cellid from raw id (0=invalid tower id) */
  HGCalDetId(uint32_t rawid);
  /** Constructor from subdetector, zplus, layer, module, cell numbers */
  HGCalDetId(ForwardSubdetector subdet, int zp, int lay, int mod, int subsec, int cell);
  /** Constructor from a generic cell id */
  HGCalDetId(const DetId& id);
  /** Assignment from a generic cell id */
  HGCalDetId& operator=(const DetId& id);

  /// get the absolute value of the cell #'s in x and y
  int cell() const { return id_&kHGCalCellMask; }

  /// get the sector #
  int sector() const { return (id_>>kHGCalSectorOffset)&kHGCalSectorMask; }

  /// get the degree subsector
  int subsector() const { return ( (id_>>kHGCalSubSectorOffset)&kHGCalSubSectorMask ? 1 : -1); }

  /// get the layer #
  int layer() const { return (id_>>kHGCalLayerOffset)&kHGCalLayerMask; }

  /// get the z-side of the cell (1/-1)
  int zside() const { return ((id_>>kHGCalZsideOffset) & kHGCalZsideMask ? 1 : -1); }

  /// consistency check : no bits left => no overhead
  bool isHGCal()   const { return true; }
  bool isForward() const { return true; }
  static bool isValid(ForwardSubdetector subdet, int zp, int lay, 
		      int mod, int subsec, int cell);

};

std::ostream& operator<<(std::ostream&,const HGCalDetId& id);

#endif
