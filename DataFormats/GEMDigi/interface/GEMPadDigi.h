#ifndef DataFormats_GEMDigi_GEMPadDigi_h
#define DataFormats_GEMDigi_GEMPadDigi_h

/** \class GEMPadDigi
 *
 * Digi for GEM-CSC trigger pads
 *  
 * \author Vadim Khotilovich
 *
 */

#include <boost/cstdint.hpp>
#include <iosfwd>

class GEMPadDigi{

public:
  explicit GEMPadDigi (int pad, int bx);
  GEMPadDigi ();

  bool operator==(const GEMPadDigi& digi) const;
  bool operator!=(const GEMPadDigi& digi) const;
  bool operator<(const GEMPadDigi& digi) const;

  int pad() const { return pad_; }
  int bx() const { return bx_; }

  void print() const;

private:
  uint16_t pad_;
  int32_t  bx_; 
};

std::ostream & operator<<(std::ostream & o, const GEMPadDigi& digi);

#endif

