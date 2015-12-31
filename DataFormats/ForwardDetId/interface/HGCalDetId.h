#ifndef DataFormats_ForwardDetId_HGCalDetId_H
#define DataFormats_ForwardDetId_HGCalDetId_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"


class HGCalDetId : public DetId {

public:
  static const int kHGCalCellOffset      = 0;
  static const int kHGCalCellMask        = 0xFF;
  static const int kHGCalWaferOffset     = 8;
  static const int kHGCalWaferMask       = 0x3FF;
  static const int kHGCalWaferTypeOffset = 18;
  static const int kHGCalWaferTypeMask   = 0x1;
  static const int kHGCalLayerOffset     = 19;
  static const int kHGCalLayerMask       = 0x1F;
  static const int kHGCalZsideOffset     = 24;
  static const int kHGCalZsideMask       = 0x1;
  static const int kHGCalMaskCell        = 0xFFFFFF00;

  /** Create a null cellid*/
  HGCalDetId();
  /** Create cellid from raw id (0=invalid tower id) */
  HGCalDetId(uint32_t rawid);
  /** Constructor from subdetector, zplus, layer, module, cell numbers */
  HGCalDetId(ForwardSubdetector subdet, int zp, int lay, int wafertype, int wafer, int cell);
  /** Constructor from a generic cell id */
  HGCalDetId(const DetId& id);
  /** Assignment from a generic cell id */
  HGCalDetId& operator=(const DetId& id);
  
  /** Converter for a geometry cell id */
  HGCalDetId geometryCell () const {return id_&kHGCalMaskCell;}

  /// get the absolute value of the cell #'s in x and y
  int cell() const { return id_&kHGCalCellMask; }

  /// get the wafer #
  int wafer() const { return (id_>>kHGCalWaferOffset)&kHGCalWaferMask; }

  /// get the wafer type
  int waferType() const { return ((id_>>kHGCalWaferTypeOffset)&kHGCalWaferTypeMask ? 1 : -1); }

  /// get the layer #
  int layer() const { return (id_>>kHGCalLayerOffset)&kHGCalLayerMask; }

  /// get the z-side of the cell (1/-1)
  int zside() const { return ((id_>>kHGCalZsideOffset) & kHGCalZsideMask ? 1 : -1); }

  /// consistency check : no bits left => no overhead
  bool isHGCal()   const { return true; }
  bool isForward() const { return true; }
  static bool isValid(ForwardSubdetector subdet, int zp, int lay, 
		      int wafertype, int wafer, int cell);

  static const HGCalDetId Undefined;
};

std::ostream& operator<<(std::ostream&,const HGCalDetId& id);

#endif
