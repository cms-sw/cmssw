#ifndef DataFormats_ForwardDetId_FastTimeDetId_H
#define DataFormats_ForwardDetId_FastTimeDetId_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"


class FastTimeDetId : public DetId {
public:
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

  /// get the subdetector
  ForwardSubdetector subdet() const { return FastTime; }

  /// get the absolute value of the cell #'s along x-axis
  int ix() const { return (id_>>8)&0xFF; }

  /// get the absolute value of the cell #'s along y-axis
  int iy() const { return id_&0xFF; }

  /// get the z-side of the cell (1/-1)
  int zside() const { return ((id_>>16) & 0x1 ? 1 : -1); }

  /// consistency check : no bits left => no overhead
  bool isFastTime() const { return true; }
  bool isForward()  const { return true; }
  
  static const FastTimeDetId Undefined;

};

std::ostream& operator<<(std::ostream&,const FastTimeDetId& id);

#endif
