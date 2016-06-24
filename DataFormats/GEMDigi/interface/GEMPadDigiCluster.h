#ifndef DataFormats_GEMDigi_GEMPadDigiCluster_h
#define DataFormats_GEMDigi_GEMPadDigiCluster_h

/** \class GEMPadDigiCluster
 *
 * Cluster of maximal 8 adjacent GEM pad digis
 * with the same BX
 *  
 * \author Sven Dildick
 *
 */

#include <boost/cstdint.hpp>
#include <iosfwd>

class GEMPadDigiCluster{

public:
  explicit GEMPadDigiCluster (int firstPad, int lastPad, int bx);
  GEMPadDigiCluster ();

  bool operator==(const GEMPadDigiCluster& digi) const;
  bool operator!=(const GEMPadDigiCluster& digi) const;
  bool operator<(const GEMPadDigiCluster& digi) const;

  int firstPad() const { return firstPad_; }
  int lastPad() const { return lastPad_; }
  int bx() const { return bx_; }

  void print() const;

private:
  uint16_t firstPad_;
  uint16_t lastPad_;
  int32_t  bx_; 
};

std::ostream & operator<<(std::ostream & o, const GEMPadDigiCluster& digi);

#endif

