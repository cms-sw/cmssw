#ifndef IOPool_Streamer_StreamDQMSerializer_h
#define IOPool_Streamer_StreamDQMSerializer_h

/**
 * StreamDQMSerializer.h
 *
 * Utility class for translating DQM objects (monitor elements)
 * into streamer message objects.
 */

#include "TBufferFile.h"

#include "IOPool/Streamer/interface/DQMEventMsgBuilder.h"
#include <cstdint>
#include <vector>

namespace edm
{

  class StreamDQMSerializer
  {

  public:

    StreamDQMSerializer();

    int serializeDQMEvent(DQMEvent::TObjectTable& toTable,
                          bool use_compression, int compression_level);

    // This object always caches the results of the last event 
    // serialization operation.  You get access to the data using the
    // following member functions.

    unsigned char* bufferPointer() const { return ptr_; }
    unsigned int currentSpaceUsed() const { return curr_space_used_; }
    unsigned int currentEventSize() const { return curr_event_size_; }
    uint32_t adler32_chksum() const { return adler32_chksum_; }

  private:

    std::vector<unsigned char> comp_buf_; // space for compressed data
    unsigned int curr_event_size_;
    unsigned int curr_space_used_; // less than curr_event_size_ if compressed
    TBufferFile rootbuf_;
    unsigned char* ptr_; // set to the place where the last event stored
    uint32_t  adler32_chksum_; // adler32 check sum for the (compressed) data

  };

}

#endif
