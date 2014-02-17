#ifndef CSCDDUStatusDigi_CSCDDUStatusDigi_h
#define CSCDDUStatusDigi_CSCDDUStatusDigi_h

/** \class CSCDDUStatusDigi
 *
 *  Digi for CSC DDU info available in DDU
 *  
 *  $Date: 2010/06/11 15:44:22 $
 *  $Revision: 1.7 $
 *
 */

#include <vector>
#include <iosfwd>
#include <stdint.h>

class CSCDDUStatusDigi{

public:

  /// Constructor for all variables 
  CSCDDUStatusDigi (const uint16_t * header, const uint16_t * trailer, uint16_t tts);

  /// Default constructor.
  CSCDDUStatusDigi () {}

  /// Data Accessors
  const uint16_t * header() const { return header_;}
  const uint16_t * trailer() const {return trailer_;}
  const uint16_t getDDUTTS() const; 
  
 /// Print the content of CSCDDUStatusDigi
 void print() const;

private:

  uint16_t header_[12];
  uint16_t trailer_[12];
  uint16_t tts_;
};

std::ostream & operator<<(std::ostream & o, const CSCDDUStatusDigi& digi);

#endif
