#ifndef CSCDDUStatusDigi_CSCDDUStatusDigi_h
#define CSCDDUStatusDigi_CSCDDUStatusDigi_h

/** \class CSCDDUStatusDigi
 *
 *  Digi for CSC DDU info available in DDU
 *  
 *  $Date: 2007/05/18 18:52:02 $
 *  $Revision: 1.1 $
 *
 */

#include <vector>
#include <boost/cstdint.hpp>

class CSCDDUStatusDigi{

public:

  /// Constructor for all variables 
  CSCDDUStatusDigi (const uint16_t * header, const uint16_t * trailer );

  /// Default constructor.
  CSCDDUStatusDigi () {}

private:

  uint16_t header_[12];
  uint16_t trailer_[12];
};

#include<iostream>
/// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const CSCDDUStatusDigi& digi) {
  o << " "; 
  o <<"\n";
 
  return o;
}

#endif
