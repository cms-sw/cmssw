/* CSCDCCHeader 9/24/04 A.Tumanov
 * documented at  http://www.physics.ohio-state.edu/~cms/ddu/ddu2.html
 */

#ifndef CSCDCCHeader_h
#define CSCDCCHeader_h

#include <string> //for bzero

class CSCDCCHeader {

 public:
  CSCDCCHeader(int bx, int l1a, int sourceId);
  CSCDCCHeader();

  int getCDFEventNumber() const; 
  int getCDFSourceId() const; 
  int getCDFFOV() const;
  int getCDFEventType() const;
  int getCDFBunchCounter() const; 
  bool check() const { return dcc_code1==0xD9 && dcc_code2==0x97;}
  unsigned short * data() {return (unsigned short *) this;}
  static unsigned sizeInWords() {return 8;}

  // gets some data filled by the event data
  friend class CSCDDUEventData;

 private:
  unsigned long long word[2];
  int BX_id, LV1_id, Source_id, dcc_code1, dcc_code2;
  
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
