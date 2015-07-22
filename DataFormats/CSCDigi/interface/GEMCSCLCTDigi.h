#ifndef CSCDigi_GEMCSCLCTDigi_h
#define CSCDigi_GEMCSCLCTDigi_h

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include <boost/cstdint.hpp>
#include <iosfwd>

class GEMCSCLCTDigi 
{
 public:
  
  /// Constructors
  GEMCSCLCTDigi(const CSCCorrelatedLCTDigi, float);
  GEMCSCLCTDigi();                               /// default

  /// return track number
  const CSCCorrelatedLCTDigi& getDigi() const { return digi_; }

  /// return bend
  int getBend()    const { return bend_; }

  ///Comparison
  bool operator == (const GEMCSCLCTDigi &) const;
  bool operator != (const GEMCSCLCTDigi &rhs) const
    { return !(this->operator==(rhs)); }

 private:
  CSCCorrelatedLCTDigi digi_;
  float bend_;
};

#endif
