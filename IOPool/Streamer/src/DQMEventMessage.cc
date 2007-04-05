/**
 * The DQMEventMsgView class is used to view the DQM data messages that
 * are exchanged between the filter units and the storage manager.
 *
 * 09-Feb-2007 - Initial Implementation
 */

#include "IOPool/Streamer/interface/DQMEventMessage.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

/**
 * Constructor.
 */
DQMEventMsgView::DQMEventMsgView(void* buf):
  buf_((uint8*)buf),head_(buf)
{
  uint8* bufPtr;
  uint32 len;

  // verify that the buffer actually contains a DQM Event message
  if (this->code() != Header::DQM_EVENT)
    {
      throw cms::Exception("MessageDecoding", "DQMEventMsgView")
        << "Invalid DQM Event message code (" << this->code()
        << "). Should be " << Header::DQM_EVENT << "\n";
    }

  // set our buffer pointer to just beyond the header
  bufPtr = buf_ + sizeof(DQMEventHeader);

  // determine the folder name
  len = convert32(bufPtr);
  bufPtr += sizeof(uint32);
  if (len >= 0)
    {
      if (len <= 10000) // prevent something totally crazy
        {
          folderName_.append((char *) bufPtr, len);
        }
      bufPtr += len;
    }

  // determine the event length and address
  eventLen_ = convert32(bufPtr);
  bufPtr += sizeof(uint32);
  eventAddr_ = bufPtr;

  // check that the event data doesn't extend beyond the reported
  // size of the message
  if ((this->headerSize() + this->eventLength()) > this->size()) {
      throw cms::Exception("MessageDecoding", "DQMEventMsgView")
        << "Inconsistent data sizes. The size of the header ("
        << this->headerSize() << ") and the data (" << this->eventLength()
        << ") exceed the size of the message (" << this->size() << ").\n";
  }
}

/**
 * Returns the run number associated with the DQM Event.
 */
uint32 DQMEventMsgView::run() const
{
  DQMEventHeader* h = (DQMEventHeader*)buf_;
  return convert32(h->runNumber_);
}

/**
 * Returns the reserved word associated with the DQM Event.
 */
uint32 DQMEventMsgView::reserved() const
{
  DQMEventHeader* h = (DQMEventHeader*)buf_;
  return convert32(h->reserved_);
}
