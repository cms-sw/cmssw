#ifndef CSCDMBTrailer2005_h
#define CSCDMBTrailer2005_h

#include "DataFormats/CSCDigi/interface/CSCDMBStatusDigi.h"
#include "EventFilter/CSCRawToDigi/interface/CSCVDMBTrailerFormat.h"

class CSCDMBHeader;

struct CSCDMBTrailer2005: public CSCVDMBTrailerFormat {
// public:
  CSCDMBTrailer2005()
    {
      bzero(data(), sizeInWords()*2);
      bits.ddu_code_1 = bits.ddu_code_2 = bits.ddu_code_3 = bits.ddu_code_4 = 0xF;
      bits.ddu_code_5 = bits.ddu_code_6 = bits.ddu_code_7 = bits.ddu_code_8 = 0xE;
    }

  CSCDMBTrailer2005(unsigned short * buf)
    {
      memcpy(data(), buf, sizeInWords()*2);
    };
/*
  CSCDMBTrailer2005(const CSCDMBStatusDigi & digi) 
    {
      memcpy(this, digi.trailer(), sizeInWords()*2);
    }
*/
  ///@@ NEEDS TO BE DONE
  void setEventInformation(const CSCDMBHeader & dmbHeader) override 
   {
      bits.dmb_id = dmbHeader.dmbID();
      bits.crate_id = dmbHeader.crateID();
      bits.dmb_l1a = dmbHeader.l1a();
      bits.dmb_bxn = dmbHeader.bxn();
   };

  unsigned crateID() const override { return bits.crate_id; };
  unsigned dmbID() const override { return bits.dmb_id; };

  unsigned dmb_l1a() const override { return bits.dmb_l1a; };
  unsigned dmb_bxn() const override { return bits.dmb_bxn; };

  unsigned alct_endtimeout() const override { return bits.alct_endtimeout; };
  unsigned tmb_endtimeout() const override { return bits.tmb_endtimeout; };
  unsigned cfeb_endtimeout() const override { return bits.cfeb_endtimeout; };

  unsigned alct_starttimeout() const override { return bits.alct_starttimeout; };
  unsigned tmb_starttimeout() const override { return bits.tmb_starttimeout; };
  unsigned cfeb_starttimeout() const override { return bits.cfeb_starttimeout; };

  unsigned cfeb_movlp() const override { return bits.cfeb_movlp; };
  unsigned dmb_l1pipe() const override { return bits.dmb_l1pipe; };

  unsigned alct_empty() const override { return bits.alct_empty; };
  unsigned tmb_empty() const override {return bits.tmb_empty; };
  unsigned cfeb_empty() const override { return bits.cfeb_empty; };

  unsigned alct_half() const override { return bits.alct_half; };
  unsigned tmb_half() const override {return bits.tmb_half; };
  unsigned cfeb_half() const override { return bits.cfeb_half; };

  unsigned alct_full() const override { return bits.alct_full; };
  unsigned tmb_full() const override {return bits.tmb_full; };
  unsigned cfeb_full() const override { return bits.cfeb_full; };

  unsigned crc22() const override { return (bits.dmb_crc_1 | (bits.dmb_crc_2 << 11)); };
  unsigned crc_lo_parity() const override { return bits.dmb_parity_1; };
  unsigned crc_hi_parity() const override { return bits.dmb_parity_2; };


  unsigned short * data() override {return (unsigned short *)(&bits);}
  unsigned short * data() const override {return (unsigned short *)(&bits);}

  bool check() const override {return bits.ddu_code_1 == 0xF && bits.ddu_code_2 == 0xF
                          && bits.ddu_code_3 == 0xF && bits.ddu_code_4 == 0xF
                          && bits.ddu_code_5 == 0xE && bits.ddu_code_6 == 0xE
                          && bits.ddu_code_7 == 0xE && bits.ddu_code_8 == 0xE;}

  unsigned sizeInWords() const override {return 8;}

  struct {
  unsigned dmb_l1a       : 8;
  unsigned dmb_bxn       : 4;  
  unsigned ddu_code_1    : 4;

  unsigned cfeb_half     : 5;
  unsigned tmb_half      : 1;
  unsigned alct_half     : 1;
  unsigned cfeb_movlp    : 5;
  unsigned ddu_code_2    : 4;

  unsigned tmb_starttimeout   : 1;
  unsigned alct_starttimeout  : 1;
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
  } bits;

};

#endif

