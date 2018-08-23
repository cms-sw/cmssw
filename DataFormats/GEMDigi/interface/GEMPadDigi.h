#ifndef DataFormats_GEMDigi_GEMPadDigi_h
#define DataFormats_GEMDigi_GEMPadDigi_h

/** \class GEMPadDigi
 *
 * Digi for GEM-CSC trigger pads
 *  
 * \author Vadim Khotilovich
 *
 */

#include <cstdint>
#include <iosfwd>

class GEMPadDigi{

public:
  explicit GEMPadDigi (int pad, int bx);
  GEMPadDigi ();

  bool operator==(const GEMPadDigi& digi) const;
  bool operator!=(const GEMPadDigi& digi) const;
  bool operator<(const GEMPadDigi& digi) const;
  bool isValid() const;

  // return the pad number. counts from 0.
  int pad() const { return pad_; }
  int bx() const { return bx_; }

  void print() const;

private:
  uint16_t pad_;
  int16_t  bx_; 
};

std::ostream & operator<<(std::ostream & o, const GEMPadDigi& digi);

#endif

