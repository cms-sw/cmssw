/**
 * This class is used to view the contents of "FED Raw Data" (FRD) events.
 * Initially, the only time that such events are created is
 * when an error occurs in the HLT.
 * So, in the beginning, "error event" == "FRD event".
 * However, if new ways of creating events of this type are added
 * to the HLT, hopefully this class can be used or upgraded to handle those
 * events as well.
 *
 * 08-Aug-2008 - KAB  - Initial Implementation
 *
 * Format:
 *   uint32 - run number
 *   uint32 - event number
 *   1024 * uint32 - size values for all 1024 FED buffers
 *   variable size - FED data
 */

#include "IOPool/Streamer/interface/FRDEventMessage.h"

/**
 * Constructor for the FRD event message viewer.
 */
FRDEventMsgView::FRDEventMsgView(void* buf): buf_((uint8*)buf)
{
  uint32* bufPtr = static_cast<uint32*>(buf);

  // for now, all we need to do here is calculate the full event size
  event_len_ = 0;
  event_len_ += sizeof(uint32);  // run number
  ++bufPtr;
  event_len_ += sizeof(uint32);  // event number
  ++bufPtr;
  for (int idx = 0; idx < 1024; idx++) {
    event_len_ += sizeof(uint32);  // FED N size
    event_len_ += *bufPtr;         // FED N data
    ++bufPtr;
  }
}

uint32 FRDEventMsgView::run() const
{
  FRDEventHeader* hdr = (FRDEventHeader*) buf_;
  return hdr->run_;
}

uint32 FRDEventMsgView::event() const
{
  FRDEventHeader* hdr = (FRDEventHeader*) buf_;
  return hdr->event_;
}
