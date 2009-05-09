#ifndef CSCDCCStatusDigi_CSCDCCStatusDigi_h
#define CSCDCCStatusDigi_CSCDCCStatusDigi_h

/** \class CSCDCCStatusDigi
 *
 *  Digi for CSC DCC info available in DDU
 *  
 *  $Date: 2008/10/29 18:34:40 $
 *  $Revision: 1.6 $
 *
 */

#include <vector>
#include <iosfwd>
#include <stdint.h>

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

std::ostream & operator<<(std::ostream & o, const CSCDCCStatusDigi& digi);

#endif
