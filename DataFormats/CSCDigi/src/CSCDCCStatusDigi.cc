/** \class CSCDCCStatusDigi
 * 
 *  Digi for CSC DCC info available in DDU
 *
 *  $Date: 2010/06/11 15:44:22 $
 *  $Revision: 1.7 $
 *
 */
#include "DataFormats/CSCDigi/interface/CSCDCCStatusDigi.h"
#include <ostream>
#include <cstring>
#include <iostream>

CSCDCCStatusDigi::CSCDCCStatusDigi(const uint16_t * header, const uint16_t * trailer, const uint32_t & error, 
                                   short unsigned tts) 
{
  errorFlag_=error;
  uint16_t headerSizeInBytes =16;
  uint16_t trailerSizeInBytes =16;
  memcpy(header_, header, headerSizeInBytes);
  memcpy(trailer_, trailer, trailerSizeInBytes);
  tts_ = tts;
}

const uint16_t CSCDCCStatusDigi::getDCCTTS() const {
         uint16_t ttsBits = (tts_ & 0x00F0) >> 4;
         return ttsBits;
}

void CSCDCCStatusDigi::print() const {
     std::cout << " Header: " << std::hex << *header_ <<
     " Trailer: " << std::hex << *trailer_ << " ErrorFlag: " <<  errorFlag_ <<
     " TTS: " << getDCCTTS() << std::dec << std::endl;
}

std::ostream & operator<<(std::ostream & o, const CSCDCCStatusDigi& digi) {
  o << " ";
  o <<"\n";

  return o;
}

