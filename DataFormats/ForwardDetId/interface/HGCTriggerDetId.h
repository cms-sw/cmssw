#ifndef DataFormats_ForwardDetId_HGCTriggerDetId_H
#define DataFormats_ForwardDetId_HGCTriggerDetId_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

class HGCTriggerDetId : public DetId {
  // |   DetId           | HGCTriggerDetId
  // | 1111     | 111    | 1     | 11111 | 111111 | 111111 | 1111111
  // | detector | subdet | zside | layer | sector | mod    | cell
  // | 15       | 7      | 2     | 31    |  64    | 64     | 127

  const static uint32_t cell_shift = 0;
  const static uint32_t cell_mask = 0x7F;
  const static uint32_t module_mask = 0x3F;
  const static uint32_t module_shift = 7;
  const static uint32_t sector_shift = 13;
  const static uint32_t sector_mask = 0x3F;
  const static uint32_t layer_shift = 19;
  const static uint32_t layer_mask = 0x1F;
  const static uint32_t zside_shift = 24;
  const static uint32_t zside_mask = 0x1;

  const inline int getMaskedId(const uint32_t& shift, const uint32_t& mask) const { return (id_ >> shift) & mask; }
  inline void setMaskedId(const uint32_t value, const uint32_t& shift, const uint32_t& mask) {
    id_ |= ((value & mask) << shift);
  }

public:
  // undefined cell, for module det id
  const static uint32_t UndefinedCell() { return cell_mask; }

  enum { Subdet = HGCTrigger };
  /** Create a null cellid*/
  HGCTriggerDetId();
  virtual ~HGCTriggerDetId() {}
  /** Create cellid from raw id (0=invalid tower id) */
  HGCTriggerDetId(uint32_t rawid);
  /** Constructor from subdetector, zplus, layer, module, cell numbers */
  HGCTriggerDetId(ForwardSubdetector subdet, int zp, int lay, int sector, int mod, int cell);
  /** Constructor from a generic cell id */
  HGCTriggerDetId(const DetId& id);
  /** Assignment from a generic cell id */
  HGCTriggerDetId& operator=(const DetId& id);

  /** Converter for a geometry cell id
   * @param id full EKDetId 
   */
  //HGCTriggerDetId geometryCell () const {return HGCTriggerDetId (subdet(), zside(), layer(), sector(), 0, 0);}

  /// get the subdetector
  ForwardSubdetector subdet() const { return HGCTrigger; }

  /// get the absolute value of the cell #'s in x and y
  int cell() const { return getMaskedId(cell_shift, cell_mask); }

  /// get the sector #
  int sector() const { return getMaskedId(sector_shift, sector_mask); }

  /// get the degree module
  int module() const { return getMaskedId(module_shift, module_mask); }

  /// get the layer #
  int layer() const { return getMaskedId(layer_shift, layer_mask); }

  /// get the z-side of the cell (1/-1)
  int zside() const { return (getMaskedId(zside_shift, zside_mask) ? 1 : -1); }

  /// consistency check : no bits left => no overhead
  bool isEE() const { return true; }
  bool isForward() const { return true; }

  static const HGCTriggerDetId Undefined;
};

std::ostream& operator<<(std::ostream&, const HGCTriggerDetId& id);

#endif
