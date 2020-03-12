#ifndef DataFormats_GEMDigi_ME0Digi_h
#define DataFormats_GEMDigi_ME0Digi_h

/** \class ME0Digi
 *
 * Digi for ME0
 *  
 * \author Sven Dildick (TAMU)
 *
 */

#include <cstdint>
#include <iosfwd>

class ME0Digi {
public:
  explicit ME0Digi(int strip, int bx);
  ME0Digi();

  bool operator==(const ME0Digi& digi) const;
  bool operator!=(const ME0Digi& digi) const;
  bool operator<(const ME0Digi& digi) const;

  // return the strip number. counts from 1.
  int strip() const { return strip_; }
  int bx() const { return bx_; }

private:
  uint16_t strip_;
  int16_t bx_;
};

std::ostream& operator<<(std::ostream& o, const ME0Digi& digi);

#endif
