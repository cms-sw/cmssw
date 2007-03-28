/**
 * This class is used to build and view the DQM Event data
 * that is exchanged between the filter units and the storage manager.
 *
 * 09-Feb-2007 - Initial Implementation
 */

#include "IOPool/Streamer/interface/DQMEventMsgBuilder.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

/**
 * Constructor.
 */
DQMEventMsgBuilder::DQMEventMsgBuilder(void* buf, uint32 bufSize, uint32 run,
                                       std::string const& folderName):
  buf_((uint8*)buf),bufSize_(bufSize)
{
  DQMEventHeader* evtHdr;
  uint8* bufPtr;
  uint32 len;

  // fill in event header information
  bufPtr = buf_ + sizeof(DQMEventHeader);
  if (((uint32) (bufPtr - buf_)) > bufSize_) {
    throw cms::Exception("MessageBuilding", "DQMEventMsgBuilder")
      << "Input buffer size is too small for required header "
      << "information.  Size = " << bufSize_
      << ", necessary size is >= "
      << ((uint32) (bufPtr - buf_)) << ".\n";
  }
  evtHdr = (DQMEventHeader*) buf_;
  convert(run, evtHdr->runNumber_);

  // copy the root folder name into the message
  len = folderName.length();
  if (((uint32) (bufPtr + len + sizeof(uint32) - buf_)) > bufSize_) {
    throw cms::Exception("MessageBuilding", "DQMEventMsgBuilder")
      << "Input buffer size is too small for required header "
      << "information.  Size = " << bufSize_
      << ", necessary size is >= "
      << ((uint32) (bufPtr + len + sizeof(uint32) - buf_)) << ".\n";
  }
  convert(len, bufPtr);
  bufPtr += sizeof(uint32);
  folderName.copy((char*) bufPtr, len);
  bufPtr += len;

  // set the event address, taking into account the size of the
  // event length field
  if (((uint32) (bufPtr + sizeof(uint32) - buf_)) > bufSize_) {
    throw cms::Exception("MessageBuilding", "DQMEventMsgBuilder")
      << "Input buffer size is too small for required header "
      << "information.  Size = " << bufSize_
      << ", necessary size is >= "
      << ((uint32) (bufPtr + sizeof(uint32) - buf_)) << ".\n";
  }
  bufPtr += sizeof(uint32);
  eventAddr_ = bufPtr;

  // set the event length to zero, initially.  The setEventLength method
  // sets the message code and message size for us.  It shouldn't be called
  // until *after* the event address is set.
  setEventLength(0);

  // initialize the reserved word to zero
  setReserved(0);
}

/**
 * Sets the value of the reserved word in the header.
 */
void DQMEventMsgBuilder::setReserved(uint32 value)
{
  DQMEventHeader* evtHdr = (DQMEventHeader*) buf_;
  convert(value, evtHdr->reserved_);
}

/**
 * Sets the length of the event (payload).  This method verifies that the
 * buffer in which we are building the message is large enough and
 * updates the size of the message taking into account the new event length.
 */
void DQMEventMsgBuilder::setEventLength(uint32 len)
{
  if (((uint32) (eventAddr_ + len - buf_)) > bufSize_) {
    throw cms::Exception("MessageBuilding", "DQMEventMsgBuilder")
      << "Event data overflows buffer. Buffer size = " << bufSize_
      << ", header size = " << this->headerSize()
      << ", event size = " << len << ".\n";
  }
  convert(len, eventAddr_ - sizeof(char_uint32));
  DQMEventHeader* evtHdr = (DQMEventHeader*) buf_;
  new (&evtHdr->header_) Header(Header::DQM_EVENT, (eventAddr_ - buf_ + len));
}

/**
 * Returns the size of the message.
 */
uint32 DQMEventMsgBuilder::size() const
{
  HeaderView v(buf_);
  return v.size();
}
