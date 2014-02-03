#include "IOPool/Streamer/interface/EOFRecordBuilder.h"
#include "IOPool/Streamer/interface/EOFRecord.h"
#include "IOPool/Streamer/interface/MsgHeader.h"
#include <cassert>
#include <cstring>

EOFRecordBuilder::EOFRecordBuilder(uint32 run, uint32 events,
                   uint32 statusCode,
                   const std::vector<uint32>& hltStats,
                   uint64 first_event_offset,
                   uint64 last_event_offset)
{
  uint32 buf_size = 1 + ((sizeof(uint32)) * 8)+ ((sizeof(uint32)) * hltStats.size());
  buf_.resize(buf_size);

  uint8* pos = (uint8*)&buf_[0];
  EOFRecordHeader* h = (EOFRecordHeader*)pos;
  convert(run,h->run_);
  convert(statusCode,h->status_code_);
  convert(events,h->events_);
  pos +=  sizeof(EOFRecordHeader);

  for(unsigned int i = 0; i < hltStats.size(); ++i) {
    char_uint32 v;
    convert(hltStats.at(i),  v);
    memcpy(pos, v, sizeof(char_uint32));
    pos += sizeof(char_uint32); 
  }

  char_uint64 v;
  convert(first_event_offset, v);
  memcpy(pos, v, sizeof(char_uint64));
  pos += sizeof(char_uint64);
  
  convert(last_event_offset, v);
  memcpy(pos, v, sizeof(char_uint64));
  pos += sizeof(char_uint64);  /** Bring the Pos at enof of Message */ 
                               /** pos -  &buf_ gives legth of message */
  uint32 calculatedSize = (uint8*)pos - (uint8*)&buf_[0];
  assert(calculatedSize == buf_.size());
  /** Code is 4 for EOF */ 
  new (&h->header_) Header(Header::EOFRECORD, calculatedSize);
}
