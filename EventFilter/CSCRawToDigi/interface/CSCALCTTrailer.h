#ifndef CSCALCTTrailer_h
#define CSCALCTTrailer_h

/** documented in  flags
  http://www.phys.ufl.edu/~madorsky/alctv/alct2000_spec.PDF
*/

#include <cstring> // memcpy
#ifndef LOCAL_UNPACK
#include <atomic>
#endif
#include "DataFormats/CSCDigi/interface/CSCALCTStatusDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

struct CSCALCTTrailer2006 {
  CSCALCTTrailer2006();

  void setFromBuffer(unsigned short const* buf) {
    memcpy(this, buf, sizeInWords()*2);
  }

  void setSize(int size) {frameCount = size;}
  void setCRC(unsigned int crc) {crc0 = crc&0x7FF; crc1 = (crc >> 11) & 0x7FF;}
  short unsigned int sizeInWords() const { ///size of ALCT Header
    return 4;
  }
  unsigned crc0:11, zero_0:1, d_0:4;
  unsigned crc1:11, zero_1:1, d_1:4;
  unsigned e0dLine:16;
  unsigned frameCount:11, reserved_3:1, d_3:4;
};

struct CSCALCTTrailer2007 {
  CSCALCTTrailer2007();

  void setFromBuffer(unsigned short const* buf) {
    memcpy(this, buf, sizeInWords()*2);
  }

  void setSize(int size) {frameCount = size;}
  void setCRC(unsigned int crc) {crc0 = crc&0x7FF; crc1 = (crc >> 11) & 0x7FF;}
  short unsigned int sizeInWords() const { ///size of ALCT Header
    return 4;
  }
  unsigned e0dLine:16;
  unsigned crc0:11, zero_0:1, reserved_0:4;
  unsigned crc1:11, zero_1:1, reserved_1:4;
  unsigned frameCount:11, reserved_3:1, reserved_4:4;
};



class CSCALCTTrailer
{
public:
  ///needed for packing
  CSCALCTTrailer(int size, int firmVersion);
  CSCALCTTrailer(const unsigned short * buf);
  CSCALCTTrailer(const CSCALCTStatusDigi & digi) {
    CSCALCTTrailer(digi.trailer());
  }

  static void setDebug(bool debugValue) {debug = debugValue;}

  unsigned short * data() {
#ifdef LOCAL_UNPACK
    switch (firmwareVersion) {
#else
    switch (firmwareVersion.load()) {
#endif
    case 2006:
      memcpy(theOriginalBuffer, &trailer2006, trailer2006.sizeInWords()*2);
      break;
    case 2007:
      memcpy(theOriginalBuffer, &trailer2007, trailer2007.sizeInWords()*2);
      break;
    default:
      edm::LogError("CSCALCTTrailer|CSCRawToDigi")
        <<"couldn't access data: ALCT firmware version is bad/not defined!";
      break;
    }
    return theOriginalBuffer;
  }

  /// in 16-bit frames
  static int sizeInWords() {return 4;}

  void setCRC(unsigned int crc){
#ifdef LOCAL_UNPACK
    switch (firmwareVersion) {
#else
    switch (firmwareVersion.load()) {
#endif
    case 2006:
      trailer2006.setCRC(crc);
      break;
    case 2007:
      trailer2007.setCRC(crc);
      break;
    default:
      edm::LogError("CSCALCTTrailer|CSCRawToDigi")
        <<"couldn't setCRC: ALCT firmware version is bad/not defined!";
      break;
    }
  }
   
    
  int getCRC() { 
#ifdef LOCAL_UNPACK
    switch (firmwareVersion) {
#else
    switch (firmwareVersion.load()) {
#endif
    case 2006:
      return ((trailer2006.crc1&0x7ff)<<11) | (trailer2006.crc0&0x7ff);
    case 2007:
      return ((trailer2007.crc1&0x7ff)<<11) | (trailer2007.crc0&0x7ff);
    default:
      edm::LogError("CSCALCTTrailer|CSCRawToDigi")
        <<"couldn't getCRC: ALCT firmware version is bad/not defined!";
      return 0;
    }
  }

  bool check() const {
#ifdef LOCAL_UNPACK
    switch (firmwareVersion) {
#else
    switch (firmwareVersion.load()) {
#endif
    case 2006:
      return (trailer2006.e0dLine & 0xfff) == 0xe0d;
    case 2007:
      return (trailer2007.e0dLine & 0xffff) == 0xde0d;
    default:
      edm::LogError("CSCALCTTrailer|CSCRawToDigi")
	<<"couldn't check: ALCT firmware version is bad/not defined!";
      return false;
    }
  }
  
  int wordCount() const {
#ifdef LOCAL_UNPACK
    switch (firmwareVersion) {
#else
    switch (firmwareVersion.load()) {
#endif
    case 2006:
      return trailer2006.frameCount;
    case 2007:
      return trailer2007.frameCount;
    default:
      edm::LogError("CSCALCTTrailer|CSCRawToDigi")
	<<"couldn't wordCount: ALCT firmware version is bad/not defined!";
      return 0;
    }
  }
  
  unsigned alctCRCCheck() const { 
#ifdef LOCAL_UNPACK
    switch (firmwareVersion) {
#else
    switch (firmwareVersion.load()) {
#endif
    case 2006:
      return trailer2006.reserved_3;
    case 2007:
      return trailer2007.reserved_3;  
    default:
      edm::LogError("CSCALCTTrailer|CSCRawToDigi")
	<<"couldn't CRCcheck: ALCT firmware version is bad/not defined!";
      return 0;
    }
  }
  
  unsigned FrameCount() const { return wordCount(); }

  CSCALCTTrailer2006 alctTrailer2006() {return trailer2006;}
  CSCALCTTrailer2007 alctTrailer2007() {return trailer2007;}

private:

#ifdef LOCAL_UNPACK
  static bool debug;
  static unsigned short int firmwareVersion;
#else
  static std::atomic<bool> debug;
  static std::atomic<unsigned short int> firmwareVersion;
#endif

  CSCALCTTrailer2006 trailer2006;
  CSCALCTTrailer2007 trailer2007;
  unsigned short int theOriginalBuffer[4];

};

#endif

