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
  virtual void setEventInformation(const CSCDMBHeader & dmbHeader) 
   {
      bits.dmb_id = dmbHeader.dmbID();
      bits.crate_id = dmbHeader.crateID();
      bits.dmb_l1a = dmbHeader.l1a();
      bits.dmb_bxn = dmbHeader.bxn();
   };

  virtual unsigned crateID() const { return bits.crate_id; };
  virtual unsigned dmbID() const { return bits.dmb_id; };

  virtual unsigned dmb_l1a() const { return bits.dmb_l1a; };
  virtual unsigned dmb_bxn() const { return bits.dmb_bxn; };

  virtual unsigned alct_endtimeout() const { return bits.alct_endtimeout; };
  virtual unsigned tmb_endtimeout() const { return bits.tmb_endtimeout; };
  virtual unsigned cfeb_endtimeout() const { return bits.cfeb_endtimeout; };

  virtual unsigned alct_starttimeout() const { return bits.alct_starttimeout; };
  virtual unsigned tmb_starttimeout() const { return bits.tmb_starttimeout; };
  virtual unsigned cfeb_starttimeout() const { return bits.cfeb_starttimeout; };

  virtual unsigned cfeb_movlp() const { return bits.cfeb_movlp; };
  virtual unsigned dmb_l1pipe() const { return bits.dmb_l1pipe; };

  virtual unsigned alct_empty() const { return bits.alct_empty; };
  virtual unsigned tmb_empty() const {return bits.tmb_empty; };
  virtual unsigned cfeb_empty() const { return bits.cfeb_empty; };

  virtual unsigned alct_half() const { return bits.alct_half; };
  virtual unsigned tmb_half() const {return bits.tmb_half; };
  virtual unsigned cfeb_half() const { return bits.cfeb_half; };

  virtual unsigned alct_full() const { return bits.alct_full; };
  virtual unsigned tmb_full() const {return bits.tmb_full; };
  virtual unsigned cfeb_full() const { return bits.cfeb_full; };

  virtual unsigned crc22() const { return (bits.dmb_crc_1 | (bits.dmb_crc_2 << 11)); };
  virtual unsigned crc_lo_parity() const { return bits.dmb_parity_1; };
  virtual unsigned crc_hi_parity() const { return bits.dmb_parity_2; };


  virtual unsigned short * data() {return (unsigned short *)(&bits);}
  virtual unsigned short * data() const {return (unsigned short *)(&bits);}

  bool check() const {return bits.ddu_code_1 == 0xF && bits.ddu_code_2 == 0xF
                          && bits.ddu_code_3 == 0xF && bits.ddu_code_4 == 0xF
                          && bits.ddu_code_5 == 0xE && bits.ddu_code_6 == 0xE
                          && bits.ddu_code_7 == 0xE && bits.ddu_code_8 == 0xE;}

  unsigned sizeInWords() const {return 8;}

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

