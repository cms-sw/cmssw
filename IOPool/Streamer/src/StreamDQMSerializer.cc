/**
 * StreamDQMSerializer.cc
 *
 * Utility class for serializing DQM objects (monitor elements)
 * into streamer message objects.
 */

#include "IOPool/Streamer/interface/ClassFiller.h"
#include "IOPool/Streamer/interface/StreamDQMSerializer.h"
#include "IOPool/Streamer/interface/StreamSerializer.h"
#include "FWCore/Utilities/interface/Adler32Calculator.h"

#include <cstdlib>

namespace edm
{

  const int init_size = 1024*1024;

  /**
   * Default constructor
   */
  StreamDQMSerializer::StreamDQMSerializer():
    comp_buf_(init_size),
    curr_event_size_(),
    curr_space_used_(),
    rootbuf_(TBuffer::kWrite,init_size),
    ptr_((unsigned char*)rootbuf_.Buffer()),
    adler32_chksum_(0)
  { }

  /**
   * Serializes the specified table of ROOT TObjects into the internal buffer.
   */
  int StreamDQMSerializer::serializeDQMEvent(DQMEvent::TObjectTable& toTable,
                                             bool use_compression,
                                             int compression_level)
  {
    // initialize the internal TBuffer
    rootbuf_.Reset();
    RootDebug tracer(10,10);

    // loop over each subfolder
    DQMEvent::TObjectTable::const_iterator sfIter;
    for (sfIter = toTable.begin(); sfIter != toTable.end(); sfIter++)
      {
        std::string folderName = sfIter->first;
        std::vector<TObject *> toList = sfIter->second;

        // serialize the ME data
        uint32 meCount = toList.size();
        for (int idx = 0; idx < (int) meCount; idx++) {
          TObject *toPtr = toList[idx];
          rootbuf_.WriteObject(toPtr);
        }
      }

    // set sizes and pointer(s) appropriately
    curr_event_size_ = rootbuf_.Length();
    curr_space_used_ = curr_event_size_;
    ptr_ = (unsigned char*) rootbuf_.Buffer();

    // compress the data, if needed
    if (use_compression)
      {
        unsigned int dest_size =
          StreamSerializer::compressBuffer(ptr_, curr_event_size_,
                                           comp_buf_, compression_level);
        // compression succeeded
        if (dest_size != 0)
          {
            ptr_ = &comp_buf_[0]; // reset to point at compressed area
            curr_space_used_ = dest_size;
          }
      }
    // calculate the adler32 checksum 
    adler32_chksum_ = cms::Adler32((char*)ptr_, curr_space_used_);


    return curr_space_used_;
  }
}
