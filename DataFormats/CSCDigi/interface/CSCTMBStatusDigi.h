#ifndef CSCTMBStatusDigi_CSCTMBStatusDigi_h
#define CSCTMBStatusDigi_CSCTMBStatusDigi_h

/** \class CSCTMBStatusDigi
 *
 *  Digi for CSC TMB info available in DDU
 *  
 *  $Date: 2007/05/22 21:04:15 $
 *  $Revision: 1.5 $
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

  /// Data Accessors
  const uint16_t * header() const {return header_;}
  const uint16_t * trailer() const {return trailer_;}

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
