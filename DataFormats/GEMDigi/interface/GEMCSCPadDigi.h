#ifndef GEMDigi_GEMCSCPadDigi_h
#define GEMDigi_GEMCSCPadDigi_h

/** \class GEMCSCPadDigi
 *
 * Digi for GEM-CSC trigger pads
 *  
 *  $Date: 2013/01/18 04:21:50 $
 *  $Revision: 1.1 $
 *
 * \author Vadim Khotilovich
 *
 */

#include <boost/cstdint.hpp>
#include <iosfwd>

class GEMCSCPadDigi{

public:
  explicit GEMCSCPadDigi (int pad, int bx);
  GEMCSCPadDigi ();

  bool operator==(const GEMCSCPadDigi& digi) const;
  bool operator<(const GEMCSCPadDigi& digi) const;

  int pad() const { return pad_; }
  int bx() const { return bx_; }

  void print() const;

private:
  uint16_t pad_;
  int32_t  bx_; 
};

std::ostream & operator<<(std::ostream & o, const GEMCSCPadDigi& digi);

#endif

