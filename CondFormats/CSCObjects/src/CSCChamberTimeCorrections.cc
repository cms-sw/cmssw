#include "CondFormats/CSCObjects/interface/CSCChamberTimeCorrections.h"
#include <iostream>

std::ostream & operator<<(std::ostream & os, const CSCChamberTimeCorrections & cscdb)
{
  for ( size_t i = 0; i < cscdb.chamberCorrections.size(); ++i )
  {
    os <<  "elem: " << i << " csc time corrections: " <<
    cscdb.chamberCorrections[i].cfeb_length         << " " <<
    cscdb.chamberCorrections[i].cfeb_rev            << " " <<
    cscdb.chamberCorrections[i].alct_length         << " " <<
    cscdb.chamberCorrections[i].alct_rev            << " " <<
    cscdb.chamberCorrections[i].cfeb_tmb_skew_delay << " " <<
    cscdb.chamberCorrections[i].cfeb_timing_corr    << " " <<
    cscdb.chamberCorrections[i].cfeb_cable_delay    << " " <<
    cscdb.chamberCorrections[i].anode_bx_offset     << "\n";
  }
  return os;
}
