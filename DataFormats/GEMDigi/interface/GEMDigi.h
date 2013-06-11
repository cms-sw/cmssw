#ifndef GEMDigi_GEMDigi_h
#define GEMDigi_GEMDigi_h

/** \class GEMDigi
 *
 * Digi for GEM
 *  
 *  $Date: 2012/12/08 01:45:22 $
 *  $Revision: 1.1 $
 *
 * \author Vadim Khotilovich
 *
 */

#include <boost/cstdint.hpp>
#include <iosfwd>

class GEMDigi{

public:
  explicit GEMDigi (int strip, int bx);
  GEMDigi ();

  bool operator==(const GEMDigi& digi) const;
  bool operator<(const GEMDigi& digi) const;

  int strip() const { return strip_; }
  int bx() const {return bx_; }

  void print() const;

private:
  uint16_t strip_;
  int32_t  bx_; 
};

std::ostream & operator<<(std::ostream & o, const GEMDigi& digi);

#endif

