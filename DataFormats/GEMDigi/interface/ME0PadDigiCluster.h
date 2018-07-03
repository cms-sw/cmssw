#ifndef DataFormats_GEMDigi_ME0PadDigiCluster_h
#define DataFormats_GEMDigi_ME0PadDigiCluster_h

/** \class ME0PadDigiCluster
 *
 * Cluster of maximal 8 adjacent ME0 pad digis
 * with the same BX
 *  
 * \author Sven Dildick
 *
 */

#include <cstdint>
#include <iosfwd>
#include <vector>

class ME0PadDigiCluster{

public:
  explicit ME0PadDigiCluster (std::vector<uint16_t> pads, int bx);
  ME0PadDigiCluster ();

  bool operator==(const ME0PadDigiCluster& digi) const;
  bool operator!=(const ME0PadDigiCluster& digi) const;
  bool operator<(const ME0PadDigiCluster& digi) const;

  const std::vector<uint16_t>& pads() const { return v_; }
  int bx() const { return bx_; }

  void print() const;

private:
  std::vector<uint16_t> v_;
  int32_t  bx_;
};

std::ostream & operator<<(std::ostream & o, const ME0PadDigiCluster& digi);

#endif

