#ifndef GEMDigi_GEMDigi_h
#define GEMDigi_GEMDigi_h

/** \class GEMDigi
 *
 * Digi for GEM
 *  
 *  $Date: 2008/10/29 18:41:18 $
 *  $Revision: 1.9 $
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

  int strip() const ;
  int bx() const;
  void print() const;

private:
  uint16_t strip_;
  int32_t  bx_; 
};

std::ostream & operator<<(std::ostream & o, const GEMDigi& digi);

#endif

