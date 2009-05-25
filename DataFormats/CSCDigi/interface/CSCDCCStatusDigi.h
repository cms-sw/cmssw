#ifndef CSCDCCStatusDigi_CSCDCCStatusDigi_h
#define CSCDCCStatusDigi_CSCDCCStatusDigi_h

/** \class CSCDCCStatusDigi
 *
 *  Digi for CSC DCC info available in DDU
 *  
 *  $Date: 2007/07/23 12:08:19 $
 *  $Revision: 1.4 $
 *
 */

#include <vector>

class CSCDCCStatusDigi{

public:

  /// Constructor for all variables 
  CSCDCCStatusDigi (const uint16_t * header, const uint16_t * trailer, 
		    const uint32_t & error);
  CSCDCCStatusDigi (const uint32_t & error) {errorFlag_=error;}
  
  /// Default constructor.
  CSCDCCStatusDigi () {}

  ///data accessors
  const uint16_t * header() const {return header_;} 
  const uint16_t * trailer() const {return trailer_;}
  const uint32_t errorFlag() const {return errorFlag_;}

private:

  uint16_t header_[8];
  uint16_t trailer_[8];
  uint32_t errorFlag_;
};

#include<iostream>
/// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const CSCDCCStatusDigi& digi) {
  o << " "; 
  o <<"\n";
 
  return o;
}

#endif
