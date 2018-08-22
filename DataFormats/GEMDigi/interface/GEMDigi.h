#ifndef DataFormats_GEMDigi_GEMDigi_h
#define DataFormats_GEMDigi_GEMDigi_h

/** \class GEMDigi
 *
 * Digi for GEM
 *  
 * \author Vadim Khotilovich
 *
 */

#include <cstdint>
#include <iosfwd>

class GEMDigi{

public:
  explicit GEMDigi (int strip, int bx);
  GEMDigi ();

  bool operator==(const GEMDigi& digi) const;
  bool operator!=(const GEMDigi& digi) const;
  bool operator<(const GEMDigi& digi) const;

  // return the strip number. counts from 0.
  int strip() const { return strip_; }
  int bx() const {return bx_; }

  void print() const;

private:
  uint16_t strip_;
  int16_t  bx_; 
};

std::ostream & operator<<(std::ostream & o, const GEMDigi& digi);

#endif

