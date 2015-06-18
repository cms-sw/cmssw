/**
 * This class is used to view the contents of "FED Raw Data" (FRD) events.
 * Initially, the only time that such events are created is
 * when an error occurs in the HLT.
 * So, in the beginning, "error event" == "FRD event".
 * However, if new ways of creating events of this type are added
 * to the HLT, hopefully this class can be used or upgraded to handle those
 * events as well.
 *
 */

#include "IOPool/Streamer/interface/FRDEventMessage.h"

/**
 * Constructor for the FRD event message viewer.
 */
FRDEventMsgView::FRDEventMsgView(void* buf)
  : buf_((uint8*)buf),
    payload_(0),
    size_(0),
    version_(0),
    run_(0),
    lumi_(0),
    event_(0),
    eventSize_(0),
    paddingSize_(0),
    adler32_(0),
    crc32c_(0)
{
  uint32* bufPtr = static_cast<uint32*>(buf);
  version_ = *bufPtr;
  // if the version number is rather large, then we assume that the true
  // version number is one.  (In version one of the format, there was
  // no version number in the data, and the run number appeared first.)
  if (version_ >= 32) {
      version_ = 1;
  }

  size_ = 0;

  // version number
  if (version_ >= 2) {
      size_ += sizeof(uint32);
      ++bufPtr;
  }

  // run number
  run_ = *bufPtr;
  size_ += sizeof(uint32);
  ++bufPtr;

  // lumi number
  if (version_ >= 2) {
      lumi_ = *bufPtr;
      size_ += sizeof(uint32);
      ++bufPtr;
  }

  // event number
  if (version_ == 4) {
    uint64 eventLow =  *bufPtr;
    size_ += sizeof(uint32);
    ++bufPtr;

    uint64 eventHigh =  *bufPtr;
    size_ += sizeof(uint32);
    ++bufPtr;

    event_ = (eventHigh << 32) | eventLow;

  } else {
    event_ = *bufPtr;
    size_ += sizeof(uint32);
    ++bufPtr;
  }

  if (version_ >= 3) {
      // event size
      eventSize_ = *bufPtr;
      size_ += sizeof(uint32) + eventSize_;
      ++bufPtr;

      if (version_ >= 5) {
        crc32c_ = *bufPtr;
        size_ += sizeof(uint32);
        ++bufPtr;
      }
      else {
        // padding size up to V4
        paddingSize_ = *bufPtr;
        size_ += sizeof(uint32) + paddingSize_;
        ++bufPtr;

        adler32_ = *bufPtr;
        size_ += sizeof(uint32);
        ++bufPtr;
      }
  }
  else {
      for (int idx = 0; idx < 1024; idx++) {
          size_ += sizeof(uint32);  // FED N size
          size_ += *bufPtr;         // FED N data
          eventSize_ += *bufPtr;
          ++bufPtr;
      }
  }

  payload_ = (void*)bufPtr;
}
