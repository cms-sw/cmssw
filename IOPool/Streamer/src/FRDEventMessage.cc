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
 * 06-Oct-2008 - KAB  - Added lumi block number
 *
 * Format:
 *   uint32 - run number
 *   uint32 - lumi number
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
  uint32 versionNumber = *bufPtr;
  // if the version number is rather large, then we assume that the true
  // version number is one.  (In version one of the format, there was
  // no version number in the data, and the run number appeared first.)
  if (versionNumber >= 32) {
      versionNumber = 1;
  }

  // for now, all we need to do here is calculate the full event size
  event_len_ = 0;
  if (versionNumber >= 2) {
      event_len_ += sizeof(uint32);  // version number
      ++bufPtr;
  }
  event_len_ += sizeof(uint32);  // run number
  ++bufPtr;
  if (versionNumber >= 2) {
      event_len_ += sizeof(uint32);  // lumi number
      ++bufPtr;
  }
  event_len_ += sizeof(uint32);  // event number
  ++bufPtr;
  for (int idx = 0; idx < 1024; idx++) {
    event_len_ += sizeof(uint32);  // FED N size
    event_len_ += *bufPtr;         // FED N data
    ++bufPtr;
  }
}

uint32 FRDEventMsgView::version() const
{
  FRDEventHeader_V2* hdr = (FRDEventHeader_V2*) buf_;
  uint32 version = hdr->version_;
  if (version >= 32) {  // value looks like run number, so assume version 1
      return 1;
  }
  else {  // version 2 and above
      return hdr->version_;
  }
}

uint32 FRDEventMsgView::run() const
{
  FRDEventHeader_V2* hdr = (FRDEventHeader_V2*) buf_;
  uint32 version = hdr->version_;
  if (version >= 32) {  // value looks like run number, so assume version 1
      FRDEventHeader_V1* hdrV1 = (FRDEventHeader_V1*) buf_;
      return hdrV1->run_;
  }
  else {  // version 2 and above
      return hdr->run_;
  }
}

uint32 FRDEventMsgView::lumi() const
{
  FRDEventHeader_V2* hdr = (FRDEventHeader_V2*) buf_;
  uint32 version = hdr->version_;
  if (version >= 32) {  // value looks like run number, so assume version 1
      return 1;
  }
  else {  // version 2 and above
      return hdr->lumi_;
  }
}

uint32 FRDEventMsgView::event() const
{
  FRDEventHeader_V2* hdr = (FRDEventHeader_V2*) buf_;
  uint32 version = hdr->version_;
  if (version >= 32) {  // value looks like run number, so assume version 1
      FRDEventHeader_V1* hdrV1 = (FRDEventHeader_V1*) buf_;
      return hdrV1->event_;
  }
  else {  // version 2 and above
      return hdr->event_;
  }
}
