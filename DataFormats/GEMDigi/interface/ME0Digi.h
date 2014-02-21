#ifndef ME0Digi_ME0Digi_h
#define ME0Digi_ME0Digi_h

/** \class ME0Digi
 *
 * Digi for ME0
 *  
 * \author Sven Dildick
 *
 */

#include <boost/cstdint.hpp>
#include <iosfwd>

class ME0Digi{

public:
  explicit ME0Digi (int strip, int bx);
  ME0Digi ();

  bool operator==(const ME0Digi& digi) const;
  bool operator<(const ME0Digi& digi) const;

  int strip() const { return strip_; }
  int bx() const {return bx_; }

  void print() const;

private:
  uint16_t strip_;
  int32_t  bx_; 
};

std::ostream & operator<<(std::ostream & o, const ME0Digi& digi);

#endif

