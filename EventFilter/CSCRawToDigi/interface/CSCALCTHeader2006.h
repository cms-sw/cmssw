#ifndef CSCRawToDigi_CSCALCTHeader2006_h
#define CSCRawToDigi_CSCALCTHeader2006_h

#include <vector>
#include <cstring>
#include <strings.h>
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
class CSCDMBHeader;

///ALCT Header consists of several modular units that are defined as structs below
struct CSCALCTHeader2006 { ///this struct contains all 2006 ALCT Header words except ALCTs
  CSCALCTHeader2006()  {
    init();
  }

  explicit CSCALCTHeader2006(int chamberType);

  void setFromBuffer(unsigned short const* buf) {
    memcpy(this, buf, sizeInWords()*2);
  }

  void init() {
     bzero(this,  sizeInWords()*2); ///size of header w/o LCTs = 8 bytes
  }

  short unsigned int sizeInWords() const { ///size of ALCT Header
    return 4;
  }

  unsigned short int BXNCount() const { return bxnCount;}

  unsigned short nLCTChipRead() const;

  void setEventInformation(const CSCDMBHeader &);///for packing

  void setDAV(int afebBoard) {activeFEBs |= 1 << afebBoard;}

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

};


struct CSCALCTs2006 {
  CSCALCTs2006() {
    bzero(this, 8); ///size of ALCT = 2bytes
  }

  void setFromBuffer(unsigned short const* buf) {
    memcpy(this, buf, sizeInWords()*2);
  }

  short unsigned int sizeInWords() const { ///size of ALCT
    return 4;
  }

  std::vector<CSCALCTDigi> ALCTDigis() const;

  /// should try to sort, but doesn't for now
  void add(const std::vector<CSCALCTDigi> & digis);
  void addALCT0(const CSCALCTDigi & digi);
  void addALCT1(const CSCALCTDigi & digi);

  /// 1st LCT lower 15 bits
  /// http://www.phys.ufl.edu/cms/emu/dqm/data_formats/ALCT_data_format_notes.pdf
  unsigned alct0_valid   : 1;
  unsigned alct0_quality : 2;
  unsigned alct0_accel   : 1;
  unsigned alct0_pattern : 1;
  unsigned alct0_key_wire: 7;
  unsigned alct0_bxn_low : 3;
  ///  DDU+LCT special word flags
  unsigned flag_4 : 1;

  unsigned alct0_bxn_high :2;
  unsigned alct0_reserved :13;
  ///  DDU+LCT special word flags
  unsigned flag_5 : 1;

  /// 2nd LCT lower 15 bits
  unsigned alct1_valid   : 1;
  unsigned alct1_quality : 2;
  unsigned alct1_accel   : 1;
  unsigned alct1_pattern : 1;
  unsigned alct1_key_wire: 7;
  unsigned alct1_bxn_low : 3;
  unsigned flag_6 : 1;

  unsigned alct1_bxn_high :2;
  unsigned alct1_reserved :13;
  unsigned flag_7 : 1;
};





#endif
