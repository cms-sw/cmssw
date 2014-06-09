#ifndef GEMDigi_GEMCSCCoPadDigi_h
#define GEMDigi_GEMCSCCoPadDigi_h

/** \class GEMCSCCoPadDigi
 *
 * Digi for GEM-CSC trigger copads
 *  
 * \author Sven Dildick
 *
 */

#include <DataFormats/GEMDigi/interface/GEMCSCPadDigi.h>
#include <boost/cstdint.hpp>
#include <iosfwd>

class GEMCSCCoPadDigi{

public:
  explicit GEMCSCCoPadDigi(GEMCSCPadDigi pad1, GEMCSCPadDigi pad2);
  GEMCSCCoPadDigi();

  bool operator==(const GEMCSCCoPadDigi& digi) const;
  bool operator!=(const GEMCSCCoPadDigi& digi) const;

  int pad(int l) const;
  int bx(int l) const;

  GEMCSCPadDigi first() const {return first_;}
  GEMCSCPadDigi second() const {return second_;}

  void print() const;

private:
  GEMCSCPadDigi first_;
  GEMCSCPadDigi second_;
};

std::ostream & operator<<(std::ostream & o, const GEMCSCCoPadDigi& digi);

#endif

