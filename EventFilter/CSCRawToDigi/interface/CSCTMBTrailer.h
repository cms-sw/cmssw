#ifndef CSCTMBTrailer_h
#define CSCTMBTrailer_h

#include <cstring> // bzero
#include "DataFormats/CSCDigi/interface/CSCTMBStatusDigi.h"
#include <cstdint>

/** Defined to begin at the 6E0C word 2006 format
6E0C
2AAA (optional)
5555 (optional)
D8+CRC22(10)
D8+CRC22(10)
DE0F
D8+WordCount
*/

/** D2007 format
6E0C
2AAA (optional)
5555 (optional)
DE0F
D8+CRC22(10)
D8+CRC22(10)
D8+WordCount
*/



class CSCTMBTrailer {
public:
  /// don't forget to pass in the size of the tmb header + clct data
  CSCTMBTrailer(int wordCount, int firmwareVersion);

  CSCTMBTrailer(unsigned short * buf, unsigned short int firmwareVersion);

  CSCTMBTrailer(const CSCTMBStatusDigi & digi)
    {
      memcpy(this, digi.trailer(), sizeInBytes());
    }

  uint16_t sizeInBytes() const {return 16;}
  unsigned int crc22() const;
  void setCRC(int crc);
  bool check() const {return theData[0]==0x6e0c;}
  /// in 16-bit frames
  int sizeInWords() const {return 5+thePadding;}
  unsigned short * data() {return theData;}

  int wordCount() const;
  static void selfTest();
private:
  int crcOffset() const  {return (theFirmwareVersion == 2006 ? 1 : 2) + thePadding;}
  int de0fOffset() const {return (theFirmwareVersion == 2006 ? 3 : 1) + thePadding;}

  unsigned short theData[7];
  int thePadding;
  unsigned short int theFirmwareVersion;
};

#endif

