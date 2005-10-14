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

class CSCDMBHeader;

class CSCALCTHeader
{
public:
  explicit CSCALCTHeader(int chamberType); 
  explicit CSCALCTHeader(const unsigned short * buf);

/** turns on the debug flag for this class */
  static void setDebug(bool value){debug = value;};



  int ALCTCRCcalc() ;
  std::bitset<22> calCRC22(const std::vector< std::bitset<16> >& datain) ;
  std::bitset<22> nextCRC22_D16(const std::bitset<16>& D, const std::bitset<22>& C);
 
 void setEventInformation(const CSCDMBHeader &);
 enum FIFO_MODE {NO_DUMP, FULL_DUMP, LOCAL_DUMP};
 unsigned short int FIFOMode()       const {return fifoMode;}
 unsigned short int NTBins()         const {return nTBins;}
 unsigned short int BoardID()        const {return boardID;}
 unsigned short int ExtTrig()        const {return extTrig;}
 unsigned short int CSCID()          const {return cscID;}
 unsigned short int BXNCount()       const {return bxnCount;}
 unsigned short int L1Acc()          const {return l1Acc;}
 unsigned short int L1AMatch()       const {return l1aMatch;}
 unsigned short int ActiveFEBs()     const {return activeFEBs;}
 unsigned short int Promote1()       const {return promote1;}
 unsigned short int Promote2()       const {return promote2;}
 unsigned short int LCTChipRead()    const {return lctChipRead;}
 unsigned short int nLCTChipRead()   const;
 
 unsigned int alct0Word() const {return LCT0_low|(LCT0_high<<15);}
 unsigned int alct1Word() const {return LCT1_low|(LCT1_high<<15);}
 
 unsigned int ALCT(const unsigned int index) const {
   if      (index == 0) return alct0Word();
   else if (index == 1) return alct1Word();
   else {
     std::cout << "+++ CSCALCTHeader:ALCT(): called with illegal index = "
	  << index << "! +++" << std::endl;
     return 0;
   }
 }

 unsigned short * data() {return (unsigned short *) this;}
 /// in 16-bit words
 static int sizeInWords() {return 8;}
 
 bool check() const {return flag_0 == 0xC;}
 
 private:
 /// l1 accept counter
 unsigned l1Acc         : 4;
 /// chamber ID number
 unsigned cscID         : 4;
 /// ALCT2000 board ID
 unsigned boardID       : 3;
 /// should be '01100', so it'll be a 6xxx in the ASCII dump
 unsigned flag_0 : 5;
 
 /// see the FIFO_MODE enum
 unsigned fifoMode : 2;
 /// # of 25 ns time bins in the raw dump
 unsigned nTBins : 5;
 /// exteran L1A arrived in L1A window
 unsigned l1aMatch : 1;
 /// trigger source was external
 unsigned extTrig : 1;
 /// promotion bit for 1st LCT pattern
 unsigned promote1 : 1;
 /// promotion bit for 2nd LCT pattern
 unsigned promote2 : 1;
 /// reserved, set to 0
 unsigned reserved_1 : 3;
 /// DDU+LCT special word flags
 unsigned flag_1 : 2;
 
  /// full bunch crossing number
  unsigned bxnCount : 12;
  /// reserved, set to 0
  unsigned reserved_2 : 2;
  ///  DDU+LCT special word flags
  unsigned flag_2 : 2;

  /// LCT chips read out in raw hit dump
  unsigned lctChipRead : 7;
  /// LCT chips with ADB hits
  unsigned activeFEBs : 7;
  ///  DDU+LCT special word flags
  unsigned flag_3 : 2;

  /// 1st LCT lower 15 bits
  unsigned LCT0_low : 15;
  ///  DDU+LCT special word flags
  unsigned flag_4 : 1;
 
  /// 1st LCT higher 15 bits
  unsigned LCT0_high : 15;
  ///  DDU+LCT special word flags
  unsigned flag_5 : 1;

  /// 2nd LCT lower 15 bits
  unsigned LCT1_low : 15;
  ///  DDU+LCT special word flags
  unsigned flag_6 : 1;

  /// 2nd LCT higher 15 bits
  unsigned LCT1_high : 15;
  ///  DDU+LCT special word flags
  unsigned flag_7 : 1;

 private:

  const unsigned short * theOriginalBuffer;

  static bool debug;
};

std::ostream & operator<<(std::ostream & os, const CSCALCTHeader & header);

#endif

