#ifndef CSCDDUStatusDigi_CSCDDUStatusDigi_h
#define CSCDDUStatusDigi_CSCDDUStatusDigi_h

/** \class CSCDDUStatusDigi
 *
 *  Digi for CSC DDU info available in DDU
 *  
 *  $Date: 2007/07/23 12:08:19 $
 *  $Revision: 1.4 $
 *
 */

#include <vector>
#include <iosfwd>

class CSCDDUStatusDigi{

public:

  /// Constructor for all variables 
  CSCDDUStatusDigi (const uint16_t * header, const uint16_t * trailer );

  /// Default constructor.
  CSCDDUStatusDigi () {}

  /// Data Accessors
  const uint16_t * header() const {return header_;}
  const uint16_t * trailer() const {return trailer_;}

private:

  uint16_t header_[12];
  uint16_t trailer_[12];
};

std::ostream & operator<<(std::ostream & o, const CSCDDUStatusDigi& digi);

#endif
