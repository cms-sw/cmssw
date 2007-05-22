// Authors Rick W and Alex T 9/24/94
//
#ifndef CSCDCCTrailer_h
#define CSCDCCTrailer_h

#include <iostream>
#include <string.h> // bzero
#include "DataFormats/CSCDigi/interface/CSCDCCStatusDigi.h"


/** documented at  http://www.physics.ohio-state.edu/~cms/ddu/ddu2.html
 */

struct CSCDCCTrailer {
  CSCDCCTrailer() 
  {
    bzero(this, sizeInWords()*2);
    dcc_trail1 = 0xEF;
  }
  
  CSCDCCTrailer(CSCDCCStatusDigi & digi)
  {
    memcpy(this, digi.trailer(), sizeInWords()*2);
  }
  /// for reference www.physics.ohio-state.edu/%7Ecms/dcc/outdatafmt.html
  /// dcc_trail1 should be EF
  unsigned dcc_trail1       : 8;
  unsigned readout_time     : 8;
  unsigned ddu_data_status1 : 16;
  unsigned ddu_data_status2 : 24;
  unsigned fifo_status      : 8;
  
  /// this line defined by CMS CDF 
  /// http://cmsdoc.cern.ch/cms/TRIDAS/horizontal/
  unsigned EOE_1        : 4;
  unsigned XXXX_1       : 4;
  unsigned Evt_lgth     : 24;
  unsigned CRC          : 16;
  unsigned XXXX_2       : 4;
  unsigned Evt_stat     : 4;
  unsigned TTS          : 4;
  unsigned Tx           : 3;
  unsigned dollardollar : 1;

  static unsigned sizeInWords() {return 8;}
  bool check() const {return dcc_trail1 == 0xEF;}
  unsigned short * data() {return (unsigned short *) this;}

};
#endif
