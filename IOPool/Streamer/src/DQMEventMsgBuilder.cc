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
DQMEventMsgBuilder::DQMEventMsgBuilder(void* buf, uint32 bufSize,
                                       uint32 run, uint32 event,
                                       std::string const& releaseTag,
                                       std::string const& topFolderName):
  buf_((uint8*)buf),bufSize_(bufSize)
{
  DQMEventHeader* evtHdr;
  uint8* bufPtr;
  uint32 len;
  uint32 protocolVersion = 1;

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
  convert(protocolVersion, evtHdr->protocolVersion_);
  convert(run, evtHdr->runNumber_);
  convert(event, evtHdr->eventNumber_);

  // copy the release tag into the message
  len = releaseTag.length();
  if (((uint32) (bufPtr + len + sizeof(uint32) - buf_)) > bufSize_) {
    throw cms::Exception("MessageBuilding", "DQMEventMsgBuilder")
      << "Input buffer size is too small for required header "
      << "information.  Size = " << bufSize_
      << ", necessary size is >= "
      << ((uint32) (bufPtr + len + sizeof(uint32) - buf_)) << ".\n";
  }
  convert(len, bufPtr);
  bufPtr += sizeof(uint32);
  releaseTag.copy((char*) bufPtr, len);
  bufPtr += len;

  // copy the top folder name into the message
  len = topFolderName.length();
  if (((uint32) (bufPtr + len + sizeof(uint32) - buf_)) > bufSize_) {
    throw cms::Exception("MessageBuilding", "DQMEventMsgBuilder")
      << "Input buffer size is too small for required header "
      << "information.  Size = " << bufSize_
      << ", necessary size is >= "
      << ((uint32) (bufPtr + len + sizeof(uint32) - buf_)) << ".\n";
  }
  convert(len, bufPtr);
  bufPtr += sizeof(uint32);
  topFolderName.copy((char*) bufPtr, len);
  bufPtr += len;

  // set the header size and the event address, taking into account the
  // size of the event length field
  if (((uint32) (bufPtr + sizeof(uint32) - buf_)) > bufSize_) {
    throw cms::Exception("MessageBuilding", "DQMEventMsgBuilder")
      << "Input buffer size is too small for required header "
      << "information.  Size = " << bufSize_
      << ", necessary size is >= "
      << ((uint32) (bufPtr + sizeof(uint32) - buf_)) << ".\n";
  }
  convert(((uint32) (bufPtr - buf_)), evtHdr->headerSize_);
  bufPtr += sizeof(uint32);
  eventAddr_ = bufPtr;

  // initialize the number of subfolders to zero
  convert((uint32) 0, bufPtr);

  // set the event length to 4 initially.  (The setEventLength method
  // sets the message code and message size for us.  It shouldn't be called
  // until *after* the event address is set.)
  setEventLength(4);

  // initialize the compression flag to zero
  setCompressionFlag(0);

  // initialize the reserved word to zero
  setReserved(0);
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
 * Adds the specified monitor element data to the DQM event.
 */
void DQMEventMsgBuilder::addMEData(std::string const& subFolderName,
                                   uint32 const monitorElementCount,
                                   TBuffer const& serializedMEData)
{
  uint8* bufPtr;
  uint32 len;

  // verify that the additional data will not overflow the buffer
  uint32 additionalSize = (3 * sizeof(uint32)) + subFolderName.length() +
    serializedMEData.BufferSize();
  if ((size() + additionalSize) > bufferSize()) {
    throw cms::Exception("MessageBuilding", "DQMEventMsgBuilder")
      << "The message buffer is full and unable to accept another "
      << "subfolder of monitor elements.  Available size = "
      << bufferSize() << ", needed size is = "
      << (size() + additionalSize) << ".\n";
  }

  // increment the number of subfolders
  bufPtr = eventAddress();
  uint32 count = convert32(bufPtr);
  count++;
  convert(count, bufPtr);

  // look up the current data size
  bufPtr -= sizeof(uint32);
  uint32 existingDataSize = convert32(bufPtr);

  // move our temporary pointer to the end of the data
  bufPtr = startAddress() + size();

  // copy the subfolder name into the message
  len = subFolderName.length();
  convert(len, bufPtr);
  bufPtr += sizeof(uint32);
  subFolderName.copy((char*) bufPtr, len);
  bufPtr += len;

  // store the number of monitor elements
  convert(monitorElementCount, bufPtr);
  bufPtr += sizeof(uint32);

  // copy the ME data into the message
  convert((uint32) serializedMEData.BufferSize(), bufPtr);
  bufPtr += sizeof(uint32);
  std::copy(serializedMEData.Buffer(),
            serializedMEData.Buffer() + serializedMEData.BufferSize(),
	    &bufPtr[0]);

  // update the event data size (and overall size as a side-effect)
  setEventLength(existingDataSize + additionalSize);
}

/**
 * Sets the value of the compression flag in the header.
 */
void DQMEventMsgBuilder::setCompressionFlag(uint32 value)
{
  DQMEventHeader* evtHdr = (DQMEventHeader*) buf_;
  convert(value, evtHdr->compressionFlag_);
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
 * Returns the size of the message.
 */
uint32 DQMEventMsgBuilder::size() const
{
  HeaderView v(buf_);
  return v.size();
}

/**
 * Returns the size of the DQM event data.
 */
uint32 DQMEventMsgBuilder::eventLength() const
{
  uint8* bufPtr = eventAddress();
  bufPtr -= sizeof(uint32);
  return convert32(bufPtr);
}
