#ifndef CSCDMBHeader_h
#define CSCDMBHeader_h

#include <cassert>
#include <iosfwd>
#include <string.h> // bzero
#include "DataFormats/CSCDigi/interface/CSCDMBStatusDigi.h"

class CSCDMBHeader  {
public:
  
  CSCDMBHeader();
  
  CSCDMBHeader(unsigned short * buf);

  CSCDMBHeader(const CSCDMBStatusDigi & digi)
    {
      memcpy(this, digi.header(), sizeInWords()*2);
    }

  bool cfebAvailable(unsigned icfeb);

  void addCFEB(int icfeb);
  void addNCLCT();  
  void addNALCT();
  void setBXN(int bxn);
  void setL1A(int l1a);
  void setCrateAddress(int crate, int dmbId);
  void setdmbID(int newDMBID) {dmb_id = newDMBID;}

  unsigned cfebActive() const {return cfeb_active;} 
  unsigned crateID() const;
  unsigned dmbID() const;
  unsigned bxn() const;
  unsigned bxn12() const;
  unsigned l1a() const;
  unsigned cfebAvailable() const;
  unsigned nalct() const;
  unsigned nclct() const;
  unsigned cfebMovlp() const;
  unsigned dmbCfebSync() const;
  unsigned activeDavMismatch() const;

  unsigned sizeInWords() const;
 
  bool check() const;

  unsigned short * data() {return (unsigned short *) this;}
  unsigned short * data() const {return (unsigned short *) this;}


  //ostream & operator<<(ostream &, const CSCDMBHeader &);

 private:

  unsigned dmb_l1a_copy2 : 12;
  /// constant, should be 1001
  unsigned newddu_code_1 : 4;

  unsigned dmb_l1a_copy1 : 12;
  /// constant, should be 1001
  unsigned newddu_code_2 : 4;

  unsigned cfeb_dav_1    : 5;
  unsigned cfeb_active   : 5;
  unsigned alct_dav_4    : 1;
  unsigned tmb_dav_4     : 1;
  /// constant, should be 1001
  unsigned newddu_code_3 : 4;


  unsigned dmb_bxn1      : 12;
  /// constant, should be 1001
  unsigned newddu_code_4 : 4;

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

  unsigned dmb_id    : 4;
  unsigned dmb_crate : 8;
  /// constant, should be '1010'
  unsigned ddu_code_2: 4;




  unsigned dmb_bxn    : 7;
  /// the time sample for this event has multiple overlaps
  /// with samples from previous events
  unsigned cfeb_movlp : 5;
  /// constant, should be '1010'
  unsigned ddu_code_3 : 4;

  unsigned dmb_l1a       : 8;
  unsigned dmb_cfeb_sync : 4;
  /// constant, should be '1010'
  unsigned ddu_code_4    : 4;


};

#endif

