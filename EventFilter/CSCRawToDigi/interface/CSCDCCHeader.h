/* CSCDCCHeader 9/24/04 A.Tumanov
 * documented at  http://www.physics.ohio-state.edu/~cms/ddu/ddu2.html
 */

#ifndef CSCDCCHeader_h
#define CSCDCCHeader_h

#include <cstdint>
#include <cstring>
#include <string> //for bzero
#include "DataFormats/CSCDigi/interface/CSCDCCStatusDigi.h"

class CSCDCCHeader {

 public:
  CSCDCCHeader(int bx, int l1a, int sourceId, int version=0);
  CSCDCCHeader();
  CSCDCCHeader(const CSCDCCStatusDigi & digi);

  void setFromBuffer(uint16_t const* buf) {
    memcpy(this, buf, sizeInWords()*2);
  }

  int getCDFEventNumber() const; 
  int getCDFSourceId() const; 
  int getCDFFOV() const;
  int getCDFEventType() const;
  int getCDFBunchCounter() const; 
  void setDAV(int dduSlot);
  bool check() const { return true/*dcc_code1==0xD9 && dcc_code2==0x97*/;}
  unsigned short * data() {return (short unsigned *)word;}
  static unsigned sizeInWords() {return 8;}

  // gets some data filled by the event data
  friend class CSCDDUEventData;

 private:
  unsigned long long word[2];
  
  /*
  //first line of DCC header definded by CDF (common data format)
  ///http://cmsdoc.cern.ch/cms/TRIDAS/horizontal/
  unsigned BOE_1       : 4;
  unsigned Evt_ty      : 4;
  unsigned LV1_id      : 24;
  unsigned BX_id       : 12;
  unsigned Source_id   : 12;
  unsigned FOV         : 4;
  unsigned Hx          : 3;
  unsigned dollardollar: 1;

  /// code1 should be D9
  /// http://www.physics.ohio-state.edu/%7Ecms/dcc/outdatafmt.html
  unsigned dcc_code1     : 8;
  unsigned orbit_counter : 32;
  unsigned fifo_status   : 16;
  ///code2 should be 97
  unsigned dcc_code2     : 8;
  
  */

};
#endif
