/* CSCDDUHeader 4/24/03 A.Tumanov
 * documented at  http://www.physics.ohio-state.edu/~cms/ddu/
 */

#ifndef CSCDDUHeader_h
#define CSCDDUHeader_h
#include "DataFormats/CSCDigi/interface/CSCDDUStatusDigi.h"
#include <cstring>
#include <cstdint>

class CSCDDUHeader {

 public:
  CSCDDUHeader();
  CSCDDUHeader(unsigned bx, unsigned l1num, unsigned sourceId, unsigned fmt_version = 0x6);
  CSCDDUHeader(const CSCDDUStatusDigi & digi)
    {
      memcpy(this, digi.header(), sizeInWords()*2);
    }

  void setFromBuffer(uint16_t const* buf) {
    memcpy(this, buf, sizeInWords()*2);
  }

  // Getters
  int s_link_status() const { return s_link_status_;}
  int format_version() const { return format_version_;}
  int source_id() const { return source_id_;}
  int bxnum() const { return bxnum_;}
  int lvl1num() const { return lvl1num_;}
  int event_type() const { return event_type_;}
  int ncsc() const { return ncsc_;}
  int dmb_dav() const { return dmb_dav_;}
  int dmb_full() const { return dmb_full_;}
  int live_cscs() const {return live_cscs_;}
  int output_path_status() const {return output_path_;}
  static unsigned sizeInWords() {return 12;}

  // Setters
  void setDMBDAV(int dduInput);
  void setSourceId(unsigned sourceId) {source_id_ = sourceId;}
  void setFormatVersion(unsigned version) {format_version_ = version & 0xF;}
  void setBXN(unsigned bxn) {bxnum_ = bxn & 0xFFF;}
  void setL1A(unsigned l1a) {lvl1num_ = l1a & 0xFFFFFF;}
  void setEventType(unsigned evt_type) {event_type_ = evt_type & 0xF;}
  void setTTSStatus(unsigned status) {tts_status_ = status & 0xF;}
  void setBOEStatus(unsigned status) {boe_status_ = status & 0x7F;}
  void setOutputPathStatus(unsigned status) {output_path_ = status & 0xFF;}

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
