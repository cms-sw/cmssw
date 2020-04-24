#ifndef CSCDMBHeader2005_h
#define CSCDMBHeader2005_h

#include <cassert>
#include <iosfwd>
#include <cstring> // bzero
#include "DataFormats/CSCDigi/interface/CSCDMBStatusDigi.h"
#include "EventFilter/CSCRawToDigi/interface/CSCVDMBHeaderFormat.h"

struct CSCDMBHeader2005: public CSCVDMBHeaderFormat  {
// public:
  
  CSCDMBHeader2005();
  
  CSCDMBHeader2005(unsigned short * buf);
/*
  CSCDMBHeader2005(const CSCDMBStatusDigi & digi)
    {
      memcpy(this, digi.header(), sizeInWords()*2);
    }
*/
  bool cfebAvailable(unsigned icfeb) override;

  void addCFEB(int icfeb) override;
  void addNCLCT() override;  
  void addNALCT() override;
  void setBXN(int bxn) override;
  void setL1A(int l1a) override;
  void setL1A24(int l1a) override;
  void setCrateAddress(int crate, int dmbId) override;
  void setdmbID(int newDMBID) override {bits.dmb_id = newDMBID;}
  void setdmbVersion(unsigned int version) override {}

  unsigned cfebActive() const override {return bits.cfeb_active;} 
  unsigned crateID() const override;
  unsigned dmbID() const override;
  unsigned bxn() const override;
  unsigned bxn12() const override;
  unsigned l1a() const override;
  unsigned l1a24() const override;
  unsigned cfebAvailable() const override;
  unsigned nalct() const override;
  unsigned nclct() const override;
  unsigned cfebMovlp() const override;
  unsigned dmbCfebSync() const override;
  unsigned activeDavMismatch() const override;
  unsigned format_version() const override;

  unsigned sizeInWords() const override;
 
  bool check() const override;

  unsigned short * data() override {return (unsigned short *)(&bits);}
  unsigned short * data() const override {return (unsigned short *)(&bits);}


  //ostream & operator<<(ostream &, const CSCDMBHeader2005 &);

// private:

  struct {

  /// 1st Header word
  unsigned dmb_l1a_lowo : 12;
  /// constant, should be 1001
  unsigned newddu_code_1 : 4;

  /// 2nd Header word
  unsigned dmb_l1a_hiwo : 12;
  /// constant, should be 1001
  unsigned newddu_code_2 : 4;

  /// 3rd Header word
  unsigned cfeb_dav_1    : 5;
  unsigned cfeb_active   : 5;
  unsigned alct_dav_4    : 1;
  unsigned tmb_dav_4     : 1;
  /// constant, should be 1001
  unsigned newddu_code_3 : 4;


  /// 4th Header word
  unsigned dmb_bxn1      : 12;
  /// constant, should be 1001
  unsigned newddu_code_4 : 4;

  /// 5th Header word
  unsigned cfeb_dav   : 5;   //5
  unsigned alct_dav_1 : 1;   // start 1
  unsigned active_dav_mismatch : 1;  // new
  unsigned tmb_dav_1  : 1;
  unsigned active_dav_mismatch_2 : 1;  // new
  unsigned alct_dav_2 : 1;
  unsigned active_dav_mismatch_3 : 1;  // new
  unsigned tmb_dav_2  : 1;
  /// constant, should be '1010'
  unsigned ddu_code_1 : 4;

  /// 6th Header word
  unsigned dmb_id    : 4;
  unsigned dmb_crate : 8;
  /// constant, should be '1010'
  unsigned ddu_code_2: 4;

  /// 7th Header word
  unsigned dmb_bxn    : 7;
  /// the time sample for this event has multiple overlaps
  /// with samples from previous events
  unsigned cfeb_movlp : 5;
  /// constant, should be '1010'
  unsigned ddu_code_3 : 4;

  /// 8th Header word
  unsigned dmb_l1a       : 8;
  unsigned dmb_cfeb_sync : 4;
  /// constant, should be '1010'
  unsigned ddu_code_4    : 4;
  } bits;


};

#endif

