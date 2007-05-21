#ifndef CSCTMBStatusDigi_CSCTMBStatusDigi_h
#define CSCTMBStatusDigi_CSCTMBStatusDigi_h

/** \class CSCTMBStatusDigi
 *
 *  Digi for CSC TMB info available in DDU
 *  
 *  $Date: 2007/05/18 18:52:02 $
 *  $Revision: 1.1 $
 *
 */

#include <vector>
#include <boost/cstdint.hpp>

class CSCTMBStatusDigi{

public:

  /// Constructor for all variables 
  CSCTMBStatusDigi (const uint16_t * header, const uint16_t * trailer );

  /// Default constructor.
  CSCTMBStatusDigi () {}

private:

  uint16_t header_[27];
  uint16_t trailer_[8];
};

#include<iostream>
/// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const CSCTMBStatusDigi& digi) {
  o << " "; 
  o <<"\n";
 
  return o;
}

#endif
