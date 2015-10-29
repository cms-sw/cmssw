#ifndef DataFormats_ForwardDetId_HGCHEDetId_H
#define DataFormats_ForwardDetId_HGCHEDetId_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"


class HGCHEDetId : public DetId {
public:
  static const int kHGCHECellOffset      = 0;
  static const int kHGCHECellMask        = 0xFFF;
  static const int kHGCHESectorOffset    = 12;
  static const int kHGCHESectorMask      = 0x3F;
  static const int kHGCHESubSectorOffset = 18;
  static const int kHGCHESubSectorMask   = 0x1;
  static const int kHGCHELayerOffset     = 19;
  static const int kHGCHELayerMask       = 0x1F;
  static const int kHGCHEZsideOffset     = 24;
  static const int kHGCHEZsideMask       = 0x1;
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
  int cell() const { return id_&kHGCHECellMask; }

  /// get the sector #
  int sector() const { return (id_>>kHGCHESectorOffset)&kHGCHESectorMask; }

  /// get the degree subsector
  int subsector() const { return ( (id_>>kHGCHESubSectorOffset)&kHGCHESubSectorMask ? 1 : -1); }

  /// get the layer #
  int layer() const { return (id_>>kHGCHELayerOffset)&kHGCHELayerMask; }

  /// get the z-side of the cell (1/-1)
  int zside() const { return ((id_>>kHGCHEZsideOffset) & kHGCHEZsideMask ? 1 : -1); }

  /// consistency check
  bool isHE()      const { return true; }
  bool isForward() const { return true; }

  static const HGCHEDetId Undefined;

};

std::ostream& operator<<(std::ostream&,const HGCHEDetId& id);

#endif

