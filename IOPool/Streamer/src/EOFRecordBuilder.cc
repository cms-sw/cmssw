#include "IOPool/Streamer/interface/EOFRecordBuilder.h"
#include "IOPool/Streamer/interface/MsgHeader.h"

EOFRecordBuilder::EOFRecordBuilder(uint32 run, uint32 events,
                   uint32 statusCode,
                   std::vector<uint32>& hltStats,
                   uint64 first_event_offset,
                   uint64 last_event_offset)
{
  cout<<"In EOFRecordBuilder"<<endl; 
  cout<<"sizeof(EOFRecordHeader):: "<<sizeof(EOFRecordHeader);
  uint32 buf_size = ((sizeof(uint32)) * 8) + sizeof(hltStats);
  cout<<"buf_size: "<<buf_size<<endl; 
  buf_.resize(buf_size);

  cout<<"Resized"<<endl;
  uint8* pos = (uint8*)&buf_[0];
  cout<<"we have pos"<<endl;
  EOFRecordHeader* h = (EOFRecordHeader*)pos;
  cout<<"Trying to add run"<<run<<endl;
  convert(run,h->run_);
  cout<<"added run"<<run<<endl;
  convert(statusCode,h->status_code_);
  cout<<"added status_code_"<<statusCode<<endl;
  convert(events,h->events_);
  cout<<"added events"<<events<<endl;
  cout<<"Still living"<<endl;

  cout<< "pos"<< pos <<endl; 
  pos +=  sizeof(EOFRecordHeader);
  cout<<"pos"<< pos <<endl; 

  for(unsigned int i=0;i<hltStats.size();++i)
    {
    char_uint32 v;
    convert(hltStats.at(i),  v);
    cout<<"hltStats.at(i)"<<hltStats.at(i)<<endl; 
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
  new (&h->header_) Header(4, (uint8*)pos - (uint8*)&buf_);  /** Code is 4 for EOF */ 
}

