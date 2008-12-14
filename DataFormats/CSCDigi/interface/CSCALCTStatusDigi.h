#ifndef CSCALCTStatusDigi_CSCALCTStatusDigi_h
#define CSCALCTStatusDigi_CSCALCTStatusDigi_h

/** \class CSCALCTStatusDigi
 *
 *  Digi for CSC ALCT info available in DDU
 *  
 *  $Date: 2007/07/23 12:08:19 $
 *  $Revision: 1.4 $
 *
 */

#include <vector>

class CSCALCTStatusDigi{

public:

  /// Constructor for all variables 
  CSCALCTStatusDigi (const uint16_t * header, const uint16_t * trailer );

  /// Default constructor.
  CSCALCTStatusDigi () {}

  /// Data Accessors
  const uint16_t * header() const {return header_;}
  const uint16_t * trailer() const {return trailer_;}



private:

  uint16_t header_[116]; ///maximum header size is 116 words
  uint16_t trailer_[4];
};

#include<iostream>
/// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const CSCALCTStatusDigi& digi) {
  o << " "; 
  o <<"\n";
 
  return o;
}

#endif
