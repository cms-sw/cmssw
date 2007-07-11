#ifndef CSCALCTHeader2007_h
#define CSCALCTHeader2007_h

#include <string.h> // memcpy
#include <iostream>
#include <iosfwd>
#include <bitset>
#include <cstdio>
#include <vector>
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTStatusDigi.h"

class CSCDMBHeader;

class CSCALCTHeader2007
{
public:
  explicit CSCALCTHeader2007(const unsigned short * buf);
  CSCALCTHeader2007(const CSCALCTStatusDigi & digi);


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
 /// 2007 firmware version by A. Madorsky
 /// bits are groupped in 16-bit words
 /// First 16 words are fixed format while the rest of the header varies in size

 unsigned wireGroupNumber_      : 3;
 unsigned backwardForward_      : 1;
 unsigned negativePositive_     : 1;
 unsigned mirrored_             : 1;
 unsigned qualityCancell_       : 1;
 unsigned latencyClocks_        : 1;
 unsigned patternB_             : 1;
 unsigned widePattern_          : 1;
 unsigned reserved_0            : 2;
 unsigned flag_0                : 4;  
   
 unsigned bxnL1A_               : 12;
 unsigned reserved_1            : 4;

 unsigned flag_1                : 16;
 
 unsigned bxnBeforeReset_       : 12;
 unsigned flag_2                : 4;
  
 unsigned bxn_                  : 12;
 unsigned rawOverflow_          : 1;
 unsigned lctOverflow_          : 1;
 unsigned configPresent_        : 1;
 unsigned flag_3                : 1;

 unsigned l1aCounter_           : 12;
 unsigned reserved_2            : 4;
 
 unsigned readoutCounter_       : 12;
 unsigned reserved_3            : 4;

 unsigned rawBins_              : 5;
 unsigned lctBins_              : 4;
 unsigned firmwareVersion_      : 6;
 unsigned flag_4                : 1;

 ///the variable size data goes here; max is 8+28+60+12=108
 unsigned short theHeaderWords[108]; 

 static bool debug;

};

std::ostream & operator<<(std::ostream & os, const CSCALCTHeader2007 & header);

#endif

