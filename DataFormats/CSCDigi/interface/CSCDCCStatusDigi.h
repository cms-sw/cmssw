#ifndef CSCDCCStatusDigi_CSCDCCStatusDigi_h
#define CSCDCCStatusDigi_CSCDCCStatusDigi_h

/** \class CSCDCCStatusDigi
 *
 *  Digi for CSC DCC info available in DDU
 *  
 *  $Date: 2010/06/11 15:44:22 $
 *  $Revision: 1.8 $
 *
 */

#include <vector>
#include <iosfwd>
#include <stdint.h>

class CSCDCCStatusDigi{

public:

  /// Constructor for all variables 
  CSCDCCStatusDigi (const uint16_t * header, const uint16_t * trailer, 
		    const uint32_t & error, short unsigned tts);
  CSCDCCStatusDigi (const uint32_t & error, short unsigned tts) {errorFlag_=error;} //tts_ = tts;}
  
  /// Default constructor.
  CSCDCCStatusDigi () {}

  ///data accessors
  const uint16_t * header() const {return header_;} 
  const uint16_t * trailer() const {return trailer_;}
  const uint32_t errorFlag() const {return errorFlag_;}
  const uint16_t getDCCTTS() const;

  /// Print the content of CSCDCCStatusDigi
 void print() const;

private:

  uint16_t header_[8];
  uint16_t trailer_[8];
  uint32_t errorFlag_;
  short unsigned tts_; /// Variable to access TTS
};

std::ostream & operator<<(std::ostream & o, const CSCDCCStatusDigi& digi);

#endif
