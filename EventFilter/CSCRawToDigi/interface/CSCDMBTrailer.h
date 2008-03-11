#ifndef CSCDMBTrailer_h
#define CSCDMBTrailer_h

#include "DataFormats/CSCDigi/interface/CSCDMBStatusDigi.h"
class CSCDMBHeader;

class CSCDMBTrailer {
public:
  CSCDMBTrailer()
    {
      bzero(this, sizeInWords()*2);
      ddu_code_1 = ddu_code_2 = ddu_code_3 = ddu_code_4 = 0xF;
      ddu_code_5 = ddu_code_6 = ddu_code_7 = ddu_code_8 = 0xE;
    }
  
  CSCDMBTrailer(const CSCDMBStatusDigi & digi) 
    {
      memcpy(this, digi.trailer(), sizeInWords()*2);
    }


  ///@@ NEEDS TO BE DONE
  void setEventInformation(const CSCDMBHeader &) {};

  unsigned short * data() {return (unsigned short *) this;}
  unsigned short * data() const {return (unsigned short *) this;}

  unsigned L1a_counter   : 8;
  unsigned dmb_bxn       : 4;  
  unsigned ddu_code_1    : 4;

  unsigned cfeb_half     : 5;
  unsigned tmb_half      : 1;
  unsigned alct_half     : 1;
  unsigned cfeb_movlp    : 5;
  unsigned ddu_code_2    : 4;

  unsigned tmb_timeout   : 1;
  unsigned alct_timeout  : 1;
  unsigned tmb_empty     : 1;
  unsigned alct_empty    : 1;
  unsigned dmb_l1pipe    : 8;
  unsigned ddu_code_3    : 4;

  unsigned cfeb_starttimeout : 5;
  unsigned tmb_endtimeout    : 1;
  unsigned alct_endtimeout   : 1; 
  unsigned cfeb_endtimeout   : 5;
  unsigned ddu_code_4        : 4;


  unsigned cfeb_empty    : 5;
  unsigned cfeb_full     : 5;
  unsigned tmb_full      : 1;
  unsigned alct_full     : 1;
  unsigned ddu_code_5    : 4;

  unsigned dmb_id        : 4; 
  unsigned crate_id      : 8;
  unsigned ddu_code_6    : 4;

  unsigned dmb_crc_1     : 11;
  unsigned dmb_parity_1  : 1;
  unsigned ddu_code_7    : 4;

  unsigned dmb_crc_2     : 11;
  unsigned dmb_parity_2  : 1;
  unsigned ddu_code_8    : 4;

  bool check() const {return ddu_code_1 == 0xF && ddu_code_2 == 0xF
                          && ddu_code_3 == 0xF && ddu_code_4 == 0xF
                          && ddu_code_5 == 0xE && ddu_code_6 == 0xE
                          && ddu_code_7 == 0xE && ddu_code_8 == 0xE;}

  static unsigned sizeInWords() {return 8;}
};

#endif

