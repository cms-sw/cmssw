#include "IOPool/Streamer/interface/EOFRecord.h"

EOFRecordView::EOFRecordView(void* buf, uint32 hlt_path_cnt):
  buf_((uint8*)buf),head_(buf),
  hlt_path_count_(hlt_path_cnt)
{
  hlt_path_stat_start_ = buf_ + sizeof(EOFRecordHeader);
  hlt_path_stat_end_ = hlt_path_stat_start_ + (hlt_path_count_*4);

  first_event_offset_ = convert64(hlt_path_stat_end_);
  last_event_offset_ = convert64(hlt_path_stat_end_ + sizeof(uint64) );
}

uint32 EOFRecordView::run() const
{
  EOFRecordHeader* h = (EOFRecordHeader*)buf_;
  return convert32(h->run_);
}

uint32 EOFRecordView::statusCode() const
{
  EOFRecordHeader* h = (EOFRecordHeader*)buf_;
  return convert32(h->status_code_);
}

uint32 EOFRecordView::events() const
{ 
  EOFRecordHeader* h = (EOFRecordHeader*)buf_;
  return convert32(h->events_);
} 

void EOFRecordView::hltStats(std::vector<uint32>& put_here) 
{
  put_here.clear();
  put_here.resize(hlt_path_count_);

  for(unsigned int i = 0; i < hlt_path_count_; ++i) {
     uint32 item = convert32(hlt_path_stat_start_ + (i*sizeof(uint32)) );
     put_here[i] = item;
  }
}


