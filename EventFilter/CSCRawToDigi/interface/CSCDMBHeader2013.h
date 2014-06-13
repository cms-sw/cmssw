#ifndef CSCDMBHeader2013_h
#define CSCDMBHeader2013_h

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


#include <cassert>
#include <iosfwd>
#include <string.h> // bzero
#include "DataFormats/CSCDigi/interface/CSCDMBStatusDigi.h"
#include "EventFilter/CSCRawToDigi/interface/CSCVDMBHeaderFormat.h"

struct CSCDMBHeader2013: public CSCVDMBHeaderFormat  {
// public:
  
  CSCDMBHeader2013();
  
  CSCDMBHeader2013(unsigned short * buf);
/*
  CSCDMBHeader2013(const CSCDMBStatusDigi & digi)
    {
      memcpy(this, digi.header(), sizeInWords()*2);
    }
*/
  virtual bool cfebAvailable(unsigned icfeb);

  virtual void addCFEB(int icfeb);
  virtual void addNCLCT();  
  virtual void addNALCT();
  virtual void setBXN(int bxn);
  virtual void setL1A(int l1a);
  virtual void setL1A24(int l1a);
  virtual void setCrateAddress(int crate, int dmbId);
  virtual void setdmbID(int newDMBID) { bits.dmb_id = newDMBID; }
  virtual void setdmbVersion(unsigned int version) {bits.fmt_version = (version<4) ? version: 0;}

  virtual unsigned cfebActive() const { return bits.cfeb_clct_sent; } 
  virtual unsigned crateID() const;
  virtual unsigned dmbID() const;
  virtual unsigned bxn() const;
  virtual unsigned bxn12() const;
  virtual unsigned l1a() const;
  virtual unsigned l1a24() const;
  virtual unsigned cfebAvailable() const;
  virtual unsigned nalct() const;
  virtual unsigned nclct() const;
  virtual unsigned cfebMovlp() const;
  virtual unsigned dmbCfebSync() const;
  virtual unsigned activeDavMismatch() const;
  virtual unsigned format_version() const;

  unsigned sizeInWords() const;
 
  bool check() const;

  virtual unsigned short * data() {return (unsigned short *)(&bits);}
  virtual unsigned short * data() const { return (unsigned short *)(&bits);}


  //ostream & operator<<(ostream &, const CSCDMBHeader &);

// private:

  struct {
  /// 1st Header word
  unsigned dmb_l1a_lowo : 12; /// DMB_L1A[11:0]   
  unsigned newddu_code_1 : 4; /// constant, should be 1001

  /// 2nd Header word
  unsigned dmb_l1a_hiwo : 12; /// DMB_L1A[23:12]
  unsigned newddu_code_2 : 4; /// constant, should be 1001

  /// 3rd Header word
  unsigned cfeb_clct_sent : 7; 	  /// CFEB_CLCT_SENT(7:1)   
  unsigned clct_dav_mismatch : 1; /// CLCT-DAV-Mismatch(1)
  unsigned fmt_version	 : 2;	  /// Fmt_Vers(1:0)
  unsigned tmb_dav       : 1;     /// TMB_DAV(1)
  unsigned alct_dav	 : 1;	  /// ALCT_DAV(1) 
  unsigned newddu_code_3 : 4; /// constant, should be 1001

  /// 4th Header word
  unsigned dmb_bxn1     : 12; /// DMB_BXN[11:0] 
  unsigned newddu_code_4 : 4; /// constant, should be 1001

  /// 5th Header word 
  unsigned cfeb_dav   : 7;    	       /// CFEB_DAV(7:1)
  unsigned clct_dav_mismatch_copy : 1; /// CLCT-DAV-Mismatch(1)
  unsigned fmt_version_copy   : 2;     /// Fmt_Vers(1:0)
  unsigned tmb_dav_copy       : 1;     /// TMB_DAV(1)
  unsigned alct_dav_copy      : 1;     /// ALCT_DAV(1) 
  unsigned ddu_code_1 : 4;   /// constant, should be '1010'

  /// 6th Header word
  unsigned dmb_id    : 4;    /// DMB_ID(4)		
  unsigned dmb_crate : 8;    /// DMB_CRATE(8)
  unsigned ddu_code_2: 4;    /// constant, should be '1010'


  /// 7th Header word
  unsigned dmb_bxn    : 5;          /// DMB_BXN[4:0]
		  /// the time sample for this event has multiple overlaps
		  /// with samples from previous events
  unsigned cfeb_movlp : 5;   	    /// CFEB_MOVLP(5:1)
  unsigned tmb_dav_copy2   : 1;     /// TMB_DAV(1)
  unsigned alct_dav_copy2  : 1;     /// ALCT_DAV(1) 
  unsigned ddu_code_3 : 4; /// constant, should be '1010'

  /// 8th Header word
  unsigned dmb_l1a       : 5;		/// DMB_L1A[4:0]
  unsigned clct_dav_mismatch_copy2 : 1; /// CLCT-DAV-Mismatch(1)
  unsigned fmt_version_copy2   : 2;     /// Fmt_Vers(1:0)
  unsigned dmb_cfeb_sync : 4;		/// DMB-CFEB-Sync[3:0]
  unsigned ddu_code_4    : 4; /// constant, should be '1010'
  } bits;


};

#endif

