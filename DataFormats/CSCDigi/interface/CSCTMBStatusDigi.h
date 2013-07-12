#ifndef CSCTMBStatusDigi_CSCTMBStatusDigi_h
#define CSCTMBStatusDigi_CSCTMBStatusDigi_h

/** \class CSCTMBStatusDigi
 *
 *  Digi for CSC TMB info available in DDU
 *  
 *  $Date: 2008/10/29 18:34:40 $
 *  $Revision: 1.9 $
 *
 */

#include <vector>
#include <iosfwd>
#include <stdint.h>

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

  uint16_t header_[43];
  uint16_t trailer_[8];
};


std::ostream & operator<<(std::ostream & o, const CSCTMBStatusDigi& digi);

#endif
