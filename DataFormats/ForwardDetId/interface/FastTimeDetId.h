#ifndef DataFormats_ForwardDetId_FastTimeDetId_H
#define DataFormats_ForwardDetId_FastTimeDetId_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"


class FastTimeDetId : public DetId {
public:
  static const int kFastTimeCellXOffset     = 8;
  static const int kFastTimeCellXMask       = 0xFF;
  static const int kFastTimeCellYOffset     = 0;
  static const int kFastTimeCellYMask       = 0xFFF;
  static const int kFastTimeZsideMask       = 0x10000;
  enum { Subdet=FastTime};
  /** Create a null cellid*/
  FastTimeDetId();
  /** Create cellid from raw id (0=invalid tower id) */
  FastTimeDetId(uint32_t rawid);
  /** Constructor from subdetector, zplus, cell numbers along x and y axes*/
  FastTimeDetId(int module_ix, int module_iy, int iz);
  /** Constructor from a generic cell id */
  FastTimeDetId(const DetId& id);
  /** Assignment from a generic cell id */
  FastTimeDetId& operator=(const DetId& id);
  
  /** Converter for a geometry cell id */
  FastTimeDetId geometryCell () const {return FastTimeDetId (0, 0, zside());}

  /// get the subdetector
  ForwardSubdetector subdet() const { return FastTime; }

  /// get the absolute value of the cell #'s along x-axis
  int ix() const { return (id_>>kFastTimeCellXOffset)&kFastTimeCellXMask; }

  /// get the absolute value of the cell #'s along y-axis
  int iy() const { return (id_>>kFastTimeCellYOffset)&kFastTimeCellYMask; }

  /// get the z-side of the cell (1/-1)
  int zside() const { return (((id_& kFastTimeZsideMask) > 0) ? 1 : -1); }

  /// consistency check : no bits left => no overhead
  bool isFastTime() const { return true; }
  bool isForward()  const { return true; }
  
  static const FastTimeDetId Undefined;

};

std::ostream& operator<<(std::ostream&,const FastTimeDetId& id);

#endif
