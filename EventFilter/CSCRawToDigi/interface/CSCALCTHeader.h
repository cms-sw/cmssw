#ifndef CSCALCTHeader_h
#define CSCALCTHeader_h

/** documented in  flags
  http://www.phys.ufl.edu/~madorsky/alctv/alct2000_spec.PDF
*/
#include <bitset>
#include <vector>
#include <iosfwd>
#ifndef LOCAL_UNPACK
#include <atomic>
#endif
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTStatusDigi.h"
#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader2006.h"
#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader2007.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <boost/dynamic_bitset.hpp>

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
#ifdef LOCAL_UNPACK
    switch (firmwareVersion) {
#else
    switch (firmwareVersion.load()) {
#endif
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
#ifdef LOCAL_UNPACK
    switch (firmwareVersion) {
#else
    switch (firmwareVersion.load()) {
#endif
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

  void setBXNCount(unsigned int bxn)       {
#ifdef LOCAL_UNPACK
    switch (firmwareVersion) {
#else
    switch (firmwareVersion.load()) {
#endif
      case 2006:
        header2006.bxnCount = bxn % 0xFFF;
	break;
      case 2007:
        header2007.bxnCount = bxn % 0xFFF;
	break;
      default:
        edm::LogError("CSCALCTHeader|CSCRawToDigi")
          <<"trying to set BXNcount: ALCT firmware version is bad/not defined!";
	break;
      }
  }


  unsigned short int L1Acc()          const {
#ifdef LOCAL_UNPACK
    switch (firmwareVersion) {
#else
    switch (firmwareVersion.load()) {
#endif
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

  void setL1Acc(unsigned int l1a)       {
#ifdef LOCAL_UNPACK
    switch (firmwareVersion) {
#else
    switch (firmwareVersion.load()) {
#endif
      case 2006:
        header2006.l1Acc = l1a % 0xF;
        break;
      case 2007:
        header2007.l1aCounter = l1a % 0xFFF;
        break;
      default:
        edm::LogError("CSCALCTHeader|CSCRawToDigi")
          <<"trying to set L1Acc: ALCT firmware version is bad/not defined!";
        break;
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
#ifdef LOCAL_UNPACK
    switch (firmwareVersion) {
#else
    switch (firmwareVersion.load()) {
#endif
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
#ifdef LOCAL_UNPACK
    switch (firmwareVersion) {
#else
    switch (firmwareVersion.load()) {
#endif
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
  static void selfTest(int firmware);  

 private:
  CSCALCTHeader2006 header2006;
  CSCALCTHeader2007 header2007;
  std::vector<CSCALCT> theALCTs;   
  CSCALCTs2006 alcts2006;
  CSCVirtexID virtexID;
  CSCConfigurationRegister configRegister;
  std::vector<CSCCollisionMask> collisionMasks;
  std::vector<CSCHotChannelMask> hotChannelMasks;
  
  //raw data also stored in this buffer
  //maximum header size is 116 words
  unsigned short int theOriginalBuffer[116];
 
#ifdef LOCAL_UNPACK
 static bool debug;
 static unsigned short int firmwareVersion;  
#else 
  static std::atomic<bool> debug;
  static std::atomic<unsigned short int> firmwareVersion;
#endif

  ///size of the 2007 header in words
  unsigned short int sizeInWords2007_, bxn0, bxn1;
};

std::ostream & operator<<(std::ostream & os, const CSCALCTHeader & header);

#endif

