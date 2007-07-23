#ifndef CSCDCCStatusDigi_CSCDCCStatusDigi_h
#define CSCDCCStatusDigi_CSCDCCStatusDigi_h

/** \class CSCDCCStatusDigi
 *
 *  Digi for CSC DCC info available in DDU
 *  
 *  $Date: 2007/05/23 18:02:50 $
 *  $Revision: 1.3 $
 *
 */

#include <vector>

class CSCDCCStatusDigi{

public:

  /// Constructor for all variables 
  CSCDCCStatusDigi (const uint16_t * header, const uint16_t * trailer );

  /// Default constructor.
  CSCDCCStatusDigi () {}

  ///data accessors
  const uint16_t * header() const {return header_;} 
  const uint16_t * trailer() const {return trailer_;}

private:

  uint16_t header_[8];
  uint16_t trailer_[8];
};

#include<iostream>
/// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const CSCDCCStatusDigi& digi) {
  o << " "; 
  o <<"\n";
 
  return o;
}

#endif
