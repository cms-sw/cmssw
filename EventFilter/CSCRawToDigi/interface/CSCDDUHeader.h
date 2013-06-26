/* CSCDDUHeader 4/24/03 A.Tumanov
 * documented at  http://www.physics.ohio-state.edu/~cms/ddu/
 */

#ifndef CSCDDUHeader_h
#define CSCDDUHeader_h
#include "DataFormats/CSCDigi/interface/CSCDDUStatusDigi.h"
#include <cstring>

class CSCDDUHeader {

 public:
  CSCDDUHeader();
  CSCDDUHeader(unsigned bx, unsigned l1num, unsigned sourceId);
  CSCDDUHeader(const CSCDDUStatusDigi & digi)
    {
      memcpy(this, digi.header(), sizeInWords()*2);
    }
  int s_link_status() const { return s_link_status_;}
  int format_version() const { return format_version_;}
  int source_id() const { return source_id_;}
  int bxnum() const { return bxnum_;}
  int lvl1num() const { return lvl1num_;}
  int event_type() const { return event_type_;}
  int ncsc() const { return ncsc_;}
  int dmb_dav() const { return dmb_dav_;}
  int dmb_full() const { return dmb_full_;}
  void setDMBDAV(int dduInput);
  int live_cscs() const {return live_cscs_;}
  int output_path_status() const {return output_path_;}
  static unsigned sizeInWords() {return 12;}
  void setSourceId(unsigned sourceId) {source_id_ = sourceId;}
  unsigned short * data() {return (unsigned short *) this;}
  bool check() const;

  // gets some data filled by the event data
  friend class CSCDDUEventData;
 private:
  /// initializes constants
  void init();
   
  unsigned s_link_status_  : 4;
  unsigned format_version_ : 4;
  unsigned source_id_      : 12;
  unsigned bxnum_          : 12;

  unsigned lvl1num_        : 24;
  unsigned event_type_     : 4;
  /// should always be 5
  unsigned bit64_          : 4;

  /// should be 8000/0001/8000
  unsigned dmb_full_  : 16;
  unsigned header2_1_ : 16;
  unsigned header2_2_ : 16;
  unsigned header2_3_ : 16;
 

  unsigned ncsc_        : 4; 
  unsigned tts_status_  : 4;
  unsigned boe_status_  : 8;
  unsigned dmb_dav_     : 16;
  unsigned output_path_ : 16;
  unsigned live_cscs_   : 16;

};
#endif
