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
#include "FWCore/Utilities/interface/Exception.h"

/**
 * Constructor for the FRD event message viewer.
 */
FRDEventMsgView::FRDEventMsgView(void* buf)
    : buf_((uint8*)buf),
      payload_(nullptr),
      size_(0),
      version_(0),
      flags_(0),
      run_(0),
      lumi_(0),
      event_(0),
      eventSize_(0),
      paddingSize_(0),
      adler32_(0),
      crc32c_(0) {
  uint32* bufPtr = static_cast<uint32*>(buf);

  // In version one of the format, there was no version number in the data,
  // and the run number (32-bit) appeared first.
  // This format is no longer supported
  version_ = *(uint16*)bufPtr;

  if (version_ < 2 || version_ > 6) {
    throw cms::Exception("FRDEventMsgView") << "FRD version " << version_ << " is not supported";
  }

  // Version 6 repurposes unused high 16-bits of 32-bit version
  // assuming we no longer need version 1 support
  flags_ = *((uint16*)bufPtr + 1);

  if (version_ < 6 && flags_) {
    throw cms::Exception("FRDEventMsgView") << "FRD flags can not be used in version " << version_;
  }

  size_ = sizeof(uint32);
  ++bufPtr;

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
    uint64 eventLow = *bufPtr;
    size_ += sizeof(uint32);
    ++bufPtr;

    uint64 eventHigh = *bufPtr;
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
    } else {
      // padding size up to V4
      paddingSize_ = *bufPtr;
      size_ += sizeof(uint32) + paddingSize_;
      ++bufPtr;

      adler32_ = *bufPtr;
      size_ += sizeof(uint32);
      ++bufPtr;
    }
  } else {
    for (int idx = 0; idx < 1024; idx++) {
      size_ += sizeof(uint32);  // FED N size
      size_ += *bufPtr;         // FED N data
      eventSize_ += *bufPtr;
      ++bufPtr;
    }
  }

  payload_ = (void*)bufPtr;
}
