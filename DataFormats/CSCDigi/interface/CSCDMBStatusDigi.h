#ifndef CSCDMBStatusDigi_CSCDMBStatusDigi_h
#define CSCDMBStatusDigi_CSCDMBStatusDigi_h

/** \class CSCDMBStatusDigi
 *
 *  Digi for CSC DMB info available in DDU
 *  
 *  $Date: 2007/05/21 20:05:07 $
 *  $Revision: 1.2 $
 *
 */

#include <vector>
#include <boost/cstdint.hpp>

class CSCDMBStatusDigi{

public:

  /// Constructor for all variables 
  CSCDMBStatusDigi (const uint16_t * header, const uint16_t * trailer );

  /// Default constructor.
  CSCDMBStatusDigi () {}

  /// Data Accessors
  uint16_t * header() {return header_;}
  uint16_t * trailer() {return trailer_;}

private:

  uint16_t header_[8];
  uint16_t trailer_[8];
};

#include<iostream>
/// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const CSCDMBStatusDigi& digi) {
  o << " "; 
  o <<"\n";
 
  return o;
}

#endif
