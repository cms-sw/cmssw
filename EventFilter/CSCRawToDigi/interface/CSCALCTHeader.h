#ifndef CSCALCTHeader_h
#define CSCALCTHeader_h

/** documented in  flags
  http://www.phys.ufl.edu/~madorsky/alctv/alct2000_spec.PDF
*/
#include <string.h> // memcpy
#include <iostream>
#include <iosfwd>
#include <bitset>
#include <cstdio>
#include <vector>
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTStatusDigi.h"
#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader2006.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <boost/dynamic_bitset.hpp>

class CSCDMBHeader;

struct CSCALCT {
  CSCALCT()  {
    bzero(this, 2); ///size of ALCT = 2bytes
  }

  CSCALCT(const CSCALCTDigi & alctDigi):
    valid(alctDigi.isValid()),
    quality(alctDigi.getQuality()),
    accel(alctDigi.getAccelerator()),
    pattern(alctDigi.getCollisionB()),
    keyWire(alctDigi.getKeyWG()),
    reserved(0)
  {
  }

  short unsigned int sizeInWords() const { ///size of ALCT
    return 1;
  }

  unsigned valid   : 1;
  unsigned quality : 2;
  unsigned accel   : 1;
  unsigned pattern : 1;
  unsigned keyWire : 7;
  unsigned reserved: 4;
};


struct CSCALCTHeader2007 {
  CSCALCTHeader2007()  {
    bzero(this,  sizeInWords()*2); ///size of 2007 header w/o variable parts = 16 bytes
    //FIXME I have no idea what these should be, so I'm just guessing
    rawBins = 16;
    lctBins = 7;
  }

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

  short unsigned int sizeInWords() const { ///size of one HotChannMask word
    return 1;
  }

  unsigned hotChannelMask  : 12;
  unsigned reserved        : 3;
  unsigned flag            : 1; ///==0
};


class CSCALCTHeader {
 public:
  explicit CSCALCTHeader(int chamberType); ///for packing

  explicit CSCALCTHeader(const unsigned short * buf);

  CSCALCTHeader(const CSCALCTStatusDigi & digi);/// to access data by via status digis

  /// turns on the debug flag for this class 
  static void setDebug(bool value){debug = value;};

  void setEventInformation(const CSCDMBHeader &);///for packing
  unsigned short int nLCTChipRead()   const;
  
  std::vector<CSCALCTDigi> ALCTDigis() const;
  
  ///some accessors here are only applicable to 2006 header
  ///some to both 2006 and 2007

  enum FIFO_MODE {NO_DUMP, FULL_DUMP, LOCAL_DUMP};
  unsigned short int FIFOMode()       const {return header2006.fifoMode;} 
  unsigned short int NTBins()         const {
    switch (firmwareVersion)
      {
      case 2006:
        return header2006.nTBins;
      case 2007:
        return header2007.rawBins;
      default:
	edm::LogError("CSCALCTHeader|CSCRawToDigi")
          <<"trying to access NTBINs: ALCT firmware version is bad/not defined!";
        return 0;
      }
  }
  unsigned short int BoardID()        const {return header2006.boardID;}
  unsigned short int ExtTrig()        const {return header2006.extTrig;}
  unsigned short int CSCID()          const {return header2006.cscID;}
  unsigned short int BXNCount()       const {
    switch (firmwareVersion)
      {
      case 2006:
        return header2006.bxnCount;
      case 2007:
        return header2007.bxnCount;
      default:
	edm::LogError("CSCALCTHeader|CSCRawToDigi")
          <<"trying to access BXNcount: ALCT firmware version is bad/not defined!";
        return 0;
      }
  }
  unsigned short int L1Acc()          const {
    switch (firmwareVersion)
      {
      case 2006:
        return header2006.l1Acc;
      case 2007:
        return header2007.l1aCounter;
      default:
	edm::LogError("CSCALCTHeader|CSCRawToDigi")
          <<"trying to access L1Acc: ALCT firmware version is bad/not defined!";
        return 0;
      }
  }
  unsigned short int L1AMatch()        const {return header2006.l1aMatch;}
  unsigned short int ActiveFEBs()      const {return header2006.activeFEBs;}
  unsigned short int Promote1()        const {return header2006.promote1;}
  unsigned short int Promote2()        const {return header2006.promote2;}
  unsigned short int LCTChipRead()     const {return header2006.lctChipRead;}
  unsigned short int alctFirmwareVersion() const {return firmwareVersion;}
  void setDAVForChannel(int wireGroup) {
     if(firmwareVersion == 2006) {
       header2006.setDAV((wireGroup-1)/16);
     }
  }
  CSCALCTHeader2007 alctHeader2007()   const {return header2007;}
  CSCALCTHeader2006 alctHeader2006()   const {return header2006;}

  unsigned short int * data() {return theOriginalBuffer;}
 
  /// in 16-bit words
  int sizeInWords() {
    switch (firmwareVersion)
      {
      case 2006:
        return 8;
      case 2007:
        return sizeInWords2007_;
      default:
	edm::LogError("CSCALCTHeader|CSCRawToDigi")
          <<"SizeInWords(): ALCT firmware version is bad/not defined!";
        return 0;
      }
  }
  
  bool check() const {
    switch (firmwareVersion)
      {
      case 2006:
	return header2006.flag_0 == 0xC;
      case 2007:
        return header2007.flag1 == 0xDB0A;
      default:
	edm::LogError("CSCALCTHeader|CSCRawToDigi")
          <<"check(): ALCT firmware version is bad/not defined!";
        return 0;
      }
  }
 
  void add(const std::vector<CSCALCTDigi> & digis);

  boost::dynamic_bitset<> pack();

  /// tests that we unpack what we packed
  static void selfTest();  

 private:

  CSCALCTHeader2007 header2007;
  CSCALCTHeader2006 header2006;
  std::vector<CSCALCT> alcts;   
  CSCALCTs2006 alcts2006;
  CSCVirtexID virtexID;
  CSCConfigurationRegister configRegister;
  std::vector<CSCCollisionMask> collisionMasks;
  std::vector<CSCHotChannelMask> hotChannelMasks;
  
  //raw data also stored in this buffer
  //maximum header size is 116 words
  unsigned short int theOriginalBuffer[116];
  
  static bool debug;
  static unsigned short int firmwareVersion;

  ///size of the 2007 header in words
  unsigned short int sizeInWords2007_, bxn0, bxn1;
};

std::ostream & operator<<(std::ostream & os, const CSCALCTHeader & header);

#endif

