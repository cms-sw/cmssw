#ifndef DataFormats_ForwardDetId_FastTimeDetId_H
#define DataFormats_ForwardDetId_FastTimeDetId_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

class FastTimeDetId : public DetId {
public:
  static const int kFastTimeCellZOffset = 10;
  static const int kFastTimeCellZMask = 0x3FF;
  static const int kFastTimeCellPhiOffset = 0;
  static const int kFastTimeCellPhiMask = 0x3FF;
  static const int kFastTimeZsideOffset = 20;
  static const int kFastTimeZsideMask = 0x1;
  static const int kFastTimeTypeOffset = 21;
  static const int kFastTimeTypeMask = 0x3;
  enum { Subdet = FastTime };
  enum { FastTimeUnknown = 0, FastTimeBarrel = 1, FastTimeEndcap = 2 };
  /** Create a null cellid*/
  FastTimeDetId();
  /** Create cellid from raw id (0=invalid tower id) */
  FastTimeDetId(uint32_t rawid);
  /** Constructor from subdetector, zplus, cell numbers along x and y axes*/
  FastTimeDetId(int type, int module_izeta, int module_iphi, int iz);
  /** Constructor from a generic cell id */
  FastTimeDetId(const DetId& id);
  /** Assignment from a generic cell id */
  FastTimeDetId& operator=(const DetId& id);

  /** Converter for a geometry cell id */
  FastTimeDetId geometryCell() const { return FastTimeDetId(type(), 0, 0, zside()); }

  /// get the subdetector
  ForwardSubdetector subdet() const { return FastTime; }

  /// get the type (barrel vs endcap)
  int type() const { return (id_ >> kFastTimeTypeOffset) & kFastTimeTypeMask; }

  /// get the absolute value of the cell #'s along x-axis (EC) | z-axis (Barel)
  int ieta() const { return (id_ >> kFastTimeCellZOffset) & kFastTimeCellZMask; }
  int iz() const { return (id_ >> kFastTimeCellZOffset) & kFastTimeCellZMask; }

  /// get the absolute value of the cell #'s along y-axis (EC) | phi (Barrel)
  int iphi() const { return (id_ >> kFastTimeCellPhiOffset) & kFastTimeCellPhiMask; }

  /// get the z-side of the cell (1/-1)
  int zside() const { return ((((id_ >> kFastTimeZsideOffset) & kFastTimeZsideMask) > 0) ? 1 : -1); }

  /// consistency check : no bits left => no overhead
  bool isFastTime() const { return true; }
  bool isForward() const { return true; }

  static const FastTimeDetId Undefined;
};

std::ostream& operator<<(std::ostream&, const FastTimeDetId& id);

#endif
