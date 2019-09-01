#ifndef DataFormats_GEMDigi_ME0PadDigi_h
#define DataFormats_GEMDigi_ME0PadDigi_h

/** \class ME0PadDigi
 *
 * Digi for ME0 trigger pads
 *  
 * \author Sven Dildick
 *
 */

#include <cstdint>
#include <iosfwd>

class ME0PadDigi {
public:
  explicit ME0PadDigi(int pad, int bx);
  ME0PadDigi();

  bool operator==(const ME0PadDigi& digi) const;
  bool operator!=(const ME0PadDigi& digi) const;
  bool operator<(const ME0PadDigi& digi) const;

  // return the pad number. counts from 1.
  int pad() const { return pad_; }
  int bx() const { return bx_; }

private:
  uint16_t pad_;
  int16_t bx_;
};

std::ostream& operator<<(std::ostream& o, const ME0PadDigi& digi);

#endif
