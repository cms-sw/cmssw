/** \class CSCDDUStatusDigi
 * 
 *  Digi for CSC DDU info available in DDU
 *
 *
 */
#include "DataFormats/CSCDigi/interface/CSCDDUStatusDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <ostream>
#include <cstring>
#include <iostream>

CSCDDUStatusDigi::CSCDDUStatusDigi(const uint16_t* header, const uint16_t* trailer, uint16_t tts) {
  uint16_t headerSizeInBytes = 24;
  uint16_t trailerSizeInBytes = 24;
  memcpy(header_, header, headerSizeInBytes);
  memcpy(trailer_, trailer, trailerSizeInBytes);
  tts_ = tts;
}

const uint16_t CSCDDUStatusDigi::getDDUTTS() const {
  uint16_t ttsBits = (tts_ & 0x00F0) >> 4;
  return ttsBits;
}

void CSCDDUStatusDigi::print() const {
  edm::LogVerbatim("CSCDigi") << " Header: " << std::hex << *header_ << " Trailer: " << std::hex << *trailer_
                              << " TTS: " << getDDUTTS() << std::dec;
}

std::ostream& operator<<(std::ostream& o, const CSCDDUStatusDigi& digi) {
  o << " ";
  o << "\n";

  return o;
}
