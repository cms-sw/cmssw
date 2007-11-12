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

class CSCDMBHeader;

class CSCALCTHeader
{
public:
  explicit CSCALCTHeader(int chamberType); 
  explicit CSCALCTHeader(const unsigned short * buf);
  CSCALCTHeader(const CSCALCTStatusDigi & digi);


/** turns on the debug flag for this class */
  static void setDebug(bool value){debug = value;};



  int ALCTCRCcalc() ;
  std::bitset<22> calCRC22(const std::vector< std::bitset<16> >& datain) ;
  std::bitset<22> nextCRC22_D16(const std::bitset<16>& D, const std::bitset<22>& C);

  std::vector<CSCALCTDigi> ALCTDigis() const; 

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
 
 /// these are decomposed into smaller words
 /// unsigned int alct0Word() const {return LCT0_low|(LCT0_high<<15);}
 /// unsigned int alct1Word() const {return LCT1_low|(LCT1_high<<15);}
 unsigned short int alct0Valid()     const {return alct0_valid;}
 unsigned short int alct0Quality()   const {return alct0_quality;}
 unsigned short int alct0Accel()     const {return alct0_accel;}
 unsigned short int alct0Pattern()   const {return alct0_pattern;}
 unsigned short int alct0KeyWire()   const {return alct0_key_wire;}
 unsigned short int alct0BXN()       const {return alct0_bxn_low|(alct0_bxn_high)<<3;}

 unsigned short int alct1Valid()     const {return alct1_valid;}
 unsigned short int alct1Quality()   const {return alct1_quality;}
 unsigned short int alct1Accel()     const {return alct1_accel;}
 unsigned short int alct1Pattern()   const {return alct1_pattern;}
 unsigned short int alct1KeyWire()   const {return alct1_key_wire;}
 unsigned short int alct1BXN()       const {return alct1_bxn_low|(alct1_bxn_high)<<3;}


 /// this is no longer supported - please use each of the words above
 /*unsigned int ALCT(const unsigned int index) const {
   if      (index == 0) return alct0Word();
   else if (index == 1) return alct1Word();
   else {
     //std::cout << "+++ CSCALCTHeader:ALCT(): called with illegal index = "
     //  << index << "! +++" << std::endl;
     return 0;
   }
 }
 */

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
  /// unsigned LCT0_low : 15;
  /// LCT0_low is decomposed into bits according to 
  /// http://www.phys.ufl.edu/cms/emu/dqm/data_formats/ALCT_data_format_notes.pdf
  unsigned alct0_valid   : 1;
  unsigned alct0_quality : 2;  
  unsigned alct0_accel   : 1;
  unsigned alct0_pattern : 1;
  unsigned alct0_key_wire: 7;
  unsigned alct0_bxn_low : 3;
  ///  DDU+LCT special word flags
  unsigned flag_4 : 1;
 
  /// 1st LCT higher 15 bits
  ///unsigned LCT0_high : 15;
  /// LCT0_high is decomposed into bits according to 
  /// http://www.phys.ufl.edu/cms/emu/dqm/data_formats/ALCT_data_format_notes.pdf
  unsigned alct0_bxn_high :2;
  unsigned alct0_reserved :13;
  ///  DDU+LCT special word flags
  unsigned flag_5 : 1;

  /// 2nd LCT lower 15 bits
  /// unsigned LCT1_low : 15;
  /// LCT1_low is decomposed into bits according to 
  /// http://www.phys.ufl.edu/cms/emu/dqm/data_formats/ALCT_data_format_notes.pdf
  unsigned alct1_valid   : 1;
  unsigned alct1_quality : 2;  
  unsigned alct1_accel   : 1;
  unsigned alct1_pattern : 1;
  unsigned alct1_key_wire: 7;
  unsigned alct1_bxn_low : 3;
  ///  DDU+LCT special word flags
  unsigned flag_6 : 1;

  /// 2nd LCT higher 15 bits
  /// unsigned LCT1_high : 15;
  /// LCT1_high is decomposed into bits according to 
  /// http://www.phys.ufl.edu/cms/emu/dqm/data_formats/ALCT_data_format_notes.pdf
  unsigned alct1_bxn_high :2;
  unsigned alct1_reserved :13;
  ///  DDU+LCT special word flags
  unsigned flag_7 : 1;

 private:

  const unsigned short * theOriginalBuffer;

  static bool debug;
};

std::ostream & operator<<(std::ostream & os, const CSCALCTHeader & header);

#endif

