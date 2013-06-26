/** EndOfFileRecord  Message Represented here

code 1 | size 4 | run 4 | status-code 4 | total events 4 |
hltbits bits * 4 | firstevent-offset 8 | 
lastevent-offset 8

*/

#ifndef IOPool_Streamer_EOFRecord_h
#define IOPool_Streamer_EOFRecord_h

#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/MsgHeader.h"

// ----------------------- eof message ------------------------

struct EOFRecordHeader
{
  Header header_;
  char_uint32 run_;
  char_uint32 status_code_;
  char_uint32 events_;
};

class EOFRecordView
{
public:
  EOFRecordView(void* buf, uint32 hlt_path_cnt);

  uint32 code() const { return head_.code(); }
  uint32 size() const { return head_.size(); }
  uint32 run() const;
  uint32 statusCode() const;
  uint32 events() const;

  uint64 firstEventOffset() {return first_event_offset_; }
  uint64 lastEventOffset() {return last_event_offset_; }

  void hltStats(std::vector<uint32>& put_here); /** Gives the total number of events that
                              were accepted for each HLT path, one
                              32-bit (4-byte) integer for each HLT path.
                              What order ???? */
private:
  uint8* buf_;
  HeaderView head_;

  uint32 hlt_path_count_;
  uint8* hlt_path_stat_start_;
  uint8* hlt_path_stat_end_;

  uint64 first_event_offset_;
  uint64 last_event_offset_;
};

#endif

