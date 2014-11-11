#ifndef CSCDMBTrailer2013_h
#define CSCDMBTrailer2013_h

/*
 DMB-DDU 2013 Header/Trailer bit definitions (some bits get repeated for bit error mitigation)

    DMB_L1A:    L1A Event Number, count from DMB
    DMB_BXN:    Bunch Crossing Number, count from DMB
    TMB_DAV:    non-zero indicates TMB data exists for this event
    ALCT_DAV:    non-zero indicates ALCT data exists for this event
    CFEB_CLCT_SENT:    indicates which CFEBs should have sent data to DMB
    CFEB_DAV:    indicates which CFEBs have sent data to DMB
    CLCT-DAV-Mismatch:    the CFEB_DAVs do not match the CLCTs sent
    DMB_CRATE:    peripheral crate ID from DMB
    DMB_ID:    board number of DMB
    CFEB_MOVLP:    the time sample for this event has multiple overlaps with samples from previous events
    DMB-CFEB-Sync:    bits used for DMB-CFEB synchronization check
    ALCT_HALF:    zero indicates that the ALCT FIFO on the DMB is half-full
    TMB_HALF:    zero indicates that the TMB FIFO on the DMB is half-full
    CFEB_HALF:    zero indicates that the CFEB FIFO on the DMB is half-full
    DMB_L1PIPE:    number of L1A Events backed-up in the DMB
    ALCT_EMPTY:    one indicates that the ALCT FIFO on the DMB is empty
    TMB_EMPTY:    one indicates that the TMB FIFO on the DMB is empty
    CFEB_EMPTY:    one indicates that the CFEB FIFO on the DMB is empty
    ALCT_Start_Timeout:    indicates that the start of ALCT data was not detected within the time-out period
    TMB_Start_Timeout:    indicates that the start of TMB data was not detected within the time-out period
    CFEB_Start_Timeout:    indicates that the start of CFEB data was not detected within the time-out period.
    CFEB_End_Timeout:    indicates that the end of CFEB data was not detected within the time-out period
    ALCT_End_Timeout:    indicates that the end of ALCT data was not detected within the time-out period
    TMB_End_Timeout:    indicates that the end of TMB data was not detected within the time-out period
    ALCT_FULL:    one indicates that the ALCT FIFO on the DMB is full
    TMB_FULL:    one indicates that the TMB FIFO on the DMB is full
    CFEB_FULL:    one indicates that the CFEB FIFO on the DMB is full
    DMB_CRC:    each DMB generates a 22-bit CRC that encompasses all CSC data from the first 9-code to the last F-code in the event
 */

#include <iostream>
#include "DataFormats/CSCDigi/interface/CSCDMBStatusDigi.h"
#include "EventFilter/CSCRawToDigi/interface/CSCVDMBTrailerFormat.h"

class CSCDMBHeader;

struct CSCDMBTrailer2013: public CSCVDMBTrailerFormat {
// public:
  CSCDMBTrailer2013()
    {
      bzero(data(), sizeInWords()*2);
      bits.ddu_code_1 = bits.ddu_code_2 = bits.ddu_code_3 = bits.ddu_code_4 = 0xF;
      bits.ddu_code_5 = bits.ddu_code_6 = bits.ddu_code_7 = bits.ddu_code_8 = 0xE;
    }
 
  CSCDMBTrailer2013(unsigned short * buf)
    {
      memcpy(data(), buf, sizeInWords()*2);
    }
  
/*  
  CSCDMBTrailer2013(const CSCDMBStatusDigi & digi) 
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

  /// Empty bits don't exists in new format
  virtual unsigned alct_empty() const { return 0; };
  virtual unsigned tmb_empty() const { return 0 ; };
  virtual unsigned cfeb_empty() const { return 0; };


  virtual unsigned alct_half() const { return bits.alct_half; };
  virtual unsigned tmb_half() const { return bits.tmb_half; };
  virtual unsigned cfeb_half() const { return bits.cfeb_half; };

  virtual unsigned alct_full() const { return bits.alct_full; };
  virtual unsigned tmb_full() const { return bits.tmb_full; };
  virtual unsigned cfeb_full() const { return (bits.cfeb_full_lowo | (bits.cfeb_full_hiwo << 3)); };
  
  virtual unsigned crc22() const { return (bits.dmb_crc_1 | (bits.dmb_crc_2 << 11)); };
  virtual unsigned crc_lo_parity() const { return bits.dmb_parity_1; };
  virtual unsigned crc_hi_parity() const { return bits.dmb_parity_2; };
 
  virtual unsigned short * data() { return (unsigned short *)(&bits); }
  virtual unsigned short * data() const { return (unsigned short *)(&bits); }

  bool check() const {return bits.ddu_code_1 == 0xF && bits.ddu_code_2 == 0xF
                          && bits.ddu_code_3 == 0xF && bits.ddu_code_4 == 0xF
                          && bits.ddu_code_5 == 0xE && bits.ddu_code_6 == 0xE
                          && bits.ddu_code_7 == 0xE && bits.ddu_code_8 == 0xE;}

  unsigned sizeInWords() const { return 8; }

  struct {
  /// 1st Trailer word
  unsigned dmb_l1a   	 : 6;  	  /// DMB_L1A[5:0]
  unsigned dmb_bxn       : 5;     /// DMB_BXN[4:0]
  unsigned alct_endtimeout : 1;   /// ALCT_End_Timeout(1) 
  unsigned ddu_code_1    : 4;     /// constant, should be '1111'

  /// 2nd Trailer word
  unsigned cfeb_endtimeout : 7;   /// CFEB_End_Timeout(7:1)
  unsigned cfeb_movlp    : 5;     /// CFEB_MOVLP(5:1)
  unsigned ddu_code_2    : 4;     /// constant, should be '1111'

  /// 3rd Trailer word
  unsigned dmb_l1pipe    : 8;     /// DMB_L1PIPE(8)
  unsigned tmb_starttimeout : 1;  /// TMB_Start_Timeout(1)
  unsigned cfeb_full_lowo : 3;    /// CFEB_FULL(3:1)
  unsigned ddu_code_3    : 4;     /// constant, should be '1111'

  /// 4th Trailer word
  unsigned cfeb_full_hiwo    : 4; /// CFEB_FULL(7:4)
  unsigned cfeb_starttimeout : 7; /// CFEB_Start_Timeout(7:1)
  unsigned alct_starttimeout : 1; /// ALCT_Start_Timeout(1) 
  unsigned ddu_code_4        : 4; /// constant, should be '1111'

  /// 5th Trailer word
  unsigned cfeb_half     : 7;   /// CFEB_HALF(7:1)
  unsigned tmb_endtimeout : 1;  /// TMB_End_Timeout(1)
  unsigned tmb_half 	 : 1;   /// TMB_HALF(1) 
  unsigned alct_half	 : 1;   /// ALCT_HALF(1)
  unsigned tmb_full      : 1;   /// TMB_FULL(1)
  unsigned alct_full     : 1;   /// ALCT_FULL(1)
  unsigned ddu_code_5    : 4;   /// constant, should be '1110'

  /// 6th Trailer word
  unsigned dmb_id        : 4;   /// DMB_ID(4)
  unsigned crate_id      : 8;   /// DMB_CRATE(8)
  unsigned ddu_code_6    : 4;   /// constant, should be '1110'

  /// 7th Trailer word
  unsigned dmb_crc_1     : 11;	/// DMB_CRC[10:0]
  unsigned dmb_parity_1  : 1;   /// DMB_CRC_LowParity(1)
  unsigned ddu_code_7    : 4;   /// constant, should be '1110'

  /// 8th Trailer word
  unsigned dmb_crc_2     : 11;	/// DMB_CRC[21:11]
  unsigned dmb_parity_2  : 1;   /// DMB_CRC_HighParity(1)
  unsigned ddu_code_8    : 4;   /// constant, should be '1110'
  } bits;

};

#endif

