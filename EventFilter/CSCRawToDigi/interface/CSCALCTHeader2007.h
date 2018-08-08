#ifndef CSCRawToDigi_CSCALCTHeader2007_h
#define CSCRawToDigi_CSCALCTHeader2007_h

/** documented in  flags
  http://www.phys.ufl.edu/~madorsky/alctv/alct2000_spec.PDF
// see http://www.phys.ufl.edu/cms/emu/dqm/data_formats/ALCT_v2p9_2008.03.28.pdf

*/
#include <bitset>
#include <vector>
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include <boost/dynamic_bitset.hpp>
#include <cstring>

class CSCDMBHeader;

struct CSCALCT {
  CSCALCT();
  CSCALCT(const CSCALCTDigi & alctDigi);

  void setFromBuffer(unsigned short const* buf) {
    memcpy(this, buf, sizeInWords()*2);
  }

  static short unsigned int sizeInWords() {return 1; }

  unsigned valid   : 1;
  unsigned quality : 2;
  unsigned accel   : 1;
  unsigned pattern : 1;
  unsigned keyWire : 7;
  unsigned reserved: 4;
};


struct CSCALCTHeader2007 {
  CSCALCTHeader2007();
  explicit CSCALCTHeader2007(int chamberType);

  void setFromBuffer(unsigned short const* buf) {
    memcpy(this, buf, sizeInWords()*2);
  }

  void setEventInformation(const CSCDMBHeader &);///for packing

  short unsigned int sizeInWords() const { ///size of ALCT2007 Header
    return 8;
  }

  unsigned flag1                : 16;///=0xDB0A

  unsigned bxnL1A               : 12;
  unsigned reserved1            : 4;

  unsigned l1aCounter           : 12;
  unsigned reserved2            : 4;

  unsigned readoutCounter       : 12;
  unsigned reserved3            : 4;

  unsigned bxnCount             : 12;
  unsigned rawOverflow          : 1;
  unsigned lctOverflow          : 1;
  unsigned configPresent        : 1;
  unsigned flag3                : 1;

  unsigned bxnBeforeReset       : 12;
  unsigned flag2                : 4;

  unsigned boardType            : 3;
  unsigned backwardForward      : 1;
  unsigned negativePositive     : 1;
  unsigned mirrored             : 1;
  unsigned qualityCancell       : 1;
  unsigned latencyClocks        : 1;
  unsigned patternB             : 1;
  unsigned widePattern          : 1;
  unsigned reserved0            : 2;
  unsigned flag0                : 4;  
   
  unsigned rawBins              : 5;
  unsigned lctBins              : 4;
  unsigned firmwareVersion      : 6;
  unsigned flag4                : 1;
};

struct CSCVirtexID {
  CSCVirtexID() {
    bzero(this,  sizeInWords()*2); ///size of virtex ID bits = 6 bytes
  }

  void setFromBuffer(unsigned short const* buf) {
    memcpy(this, buf, sizeInWords()*2);
  }

  short unsigned int sizeInWords() const { ///size of VirtexID
    return 3;
  }

  unsigned virtexIDLow  : 15;
  unsigned flag0        : 1; ///==0
  
  unsigned virtexIDMed  : 15;
  unsigned flag1        : 1; ///==0

  unsigned virtexIDHigh : 10; 
  unsigned trReg        : 3;
  unsigned reserved     : 2;
  unsigned flag2        : 1; ///==0
};

struct CSCConfigurationRegister {
  CSCConfigurationRegister()  {
    bzero(this, sizeInWords()*2); 
  }

  void setFromBuffer(unsigned short const* buf) {
    memcpy(this, buf, sizeInWords()*2);
  }

  short unsigned int sizeInWords() const { ///size of ConfigReg
    return 5;
  }


  unsigned configRegister0  : 15;
  unsigned flag0            : 1; ///==0

  unsigned configRegister1  : 15;
  unsigned flag1            : 1; ///==0

  unsigned configRegister2  : 15;
  unsigned flag2            : 1; ///==0

  unsigned configRegister3  : 15;
  unsigned flag3            : 1; ///==0

  unsigned configRegister4  : 9;
  unsigned reserved         : 6;
  unsigned flag4            : 1; ///==0
};

struct CSCCollisionMask {
  CSCCollisionMask()  {
    bzero(this, sizeInWords()*2);
  }

  void setFromBuffer(unsigned short const* buf) {
    memcpy(this, buf, sizeInWords()*2);
  }

  short unsigned int sizeInWords() const { ///size of one CollMask word
    return 1;
  }

  unsigned collisionMaskRegister  : 14;
  unsigned reserved               : 1;
  unsigned flag                   : 1; ///==0
};

struct CSCHotChannelMask {
  CSCHotChannelMask()  {
    bzero(this, sizeInWords()*2);
  }

  void setFromBuffer(unsigned short const* buf) {
    memcpy(this, buf, sizeInWords()*2);
  }

  short unsigned int sizeInWords() const { ///size of one HotChannMask word
    return 1;
  }

  unsigned hotChannelMask  : 12;
  unsigned reserved        : 3;
  unsigned flag            : 1; ///==0
};

#endif
