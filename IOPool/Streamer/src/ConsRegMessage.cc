/**
 * These classes are used to build and view
 * the registration requests and replies that are exchanged between
 * individual event consumers and the event server.
 *
 * 15-Aug-2006 - KAB  - Initial Implementation
 */

#include "IOPool/Streamer/interface/ConsRegMessage.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cassert>

/**
 * Constructor for the consumer registration request builder.
 */
ConsRegRequestBuilder::ConsRegRequestBuilder(void* buf, uint32 bufSize,
                                             std::string const& consumerName,
                                             std::string const& requestParamSet):
  buf_((uint8*)buf),bufSize_(bufSize)
{
  // update the buffer pointer to just beyond the header
  uint8* bufPtr = buf_ + sizeof(Header);
  //std::cout << "bufPtr = 0x" << hex << ((uint32) bufPtr) << dec << std::endl;
  //std::cout << "buf_ = 0x" << hex << ((uint32) buf_) << dec << std::endl;
  //std::cout << "bufSize_ = " << bufSize_ << std::endl;
  assert(((uint32) (bufPtr - buf_)) <= bufSize_);

  // copy the consumer name into the message
  uint32 len = consumerName.length();
  assert(((uint32) (bufPtr + len + sizeof(uint32) - buf_)) <= bufSize_);
  convert(len, bufPtr);
  bufPtr += sizeof(uint32);
  consumerName.copy((char *) bufPtr, len);
  bufPtr += len;

  // copy the request parameter set into the message
  len = requestParamSet.length();
  assert(((uint32) (bufPtr + len + sizeof(uint32) - buf_)) <= bufSize_);
  convert(len, bufPtr);
  bufPtr += sizeof(uint32);
  requestParamSet.copy((char *) bufPtr, len);
  bufPtr += len;

  // create the message header now that we now the full size
  //std::cout << "bufPtr = 0x" << hex << ((uint32) bufPtr) << dec << std::endl;
  //std::cout << "buf_ = 0x" << hex << ((uint32) buf_) << dec << std::endl;
  new (buf_) Header(Header::CONS_REG_REQUEST, (bufPtr - buf_));
}

/**
 * Returns the size of the consumer registration request.
 */
uint32 ConsRegRequestBuilder::size() const
{
  HeaderView hview(buf_);
  return hview.size();
}

/**
 * Constructor for the consumer registration request viewer.
 */
ConsRegRequestView::ConsRegRequestView(void* buf):
  buf_((uint8*)buf),head_(buf)
{
  // verify that the buffer actually contains a registration request
  if (this->code() != Header::CONS_REG_REQUEST)
    {
      throw cms::Exception("MessageDecoding","ConsRegRequestView")
        << "Invalid consumer registration request message code ("
        << this->code() << "). Should be " << Header::CONS_REG_REQUEST << "\n";
    }

  // update the buffer pointer to just beyond the header
  uint8* bufPtr = buf_ + sizeof(Header);

  // determine the consumer name
  uint32 len = convert32(bufPtr);
  bufPtr += sizeof(uint32);
  if (len <= 256) // len >= 0, since len is unsigned
  {
    consumerName_.append((char *) bufPtr, len);
  }
  bufPtr += len;

  // determine the request parameter set (maintain backward compatibility
  // with sources of registration requests that don't have the param set)
  if (bufPtr < (buf_ + this->size()))
  {
      len = convert32(bufPtr);
      bufPtr += sizeof(uint32);
      // what is a reasonable limit?  This is just to prevent
      // a bogus, really large value from being used...
      if (len <= 65000) // len >= 0, since len is unsigned
      {
        requestParameterSet_.append((char *) bufPtr, len);
      }
      bufPtr += len;
      assert(bufPtr); // silence clang static analyzer
  }
}

/**
 * Constructor for the consumer registration response builder.
 */
ConsRegResponseBuilder::ConsRegResponseBuilder(void* buf, uint32 bufSize,
                                               uint32 status,
                                               uint32 consumerId):
  buf_((uint8*)buf),bufSize_(bufSize)
{
  // update the buffer pointer to just beyond the header
  uint8* bufPtr = buf_ + sizeof(Header);
  assert(((uint32) (bufPtr - buf_)) <= bufSize_);

  // encode the status
  assert(((uint32) (bufPtr + sizeof(uint32) - buf_)) <= bufSize_);
  convert (status, bufPtr);
  bufPtr += sizeof(uint32);

  // encode the consumer ID
  assert(((uint32) (bufPtr + sizeof(uint32) - buf_)) <= bufSize_);
  convert (consumerId, bufPtr);
  bufPtr += sizeof(uint32);

  // create the message header now that we now the full size
  new (buf_) Header(Header::CONS_REG_RESPONSE, (bufPtr - buf_));
}

/**
 * Returns the size of the consumer registration response.
 */
uint32 ConsRegResponseBuilder::size() const
{
  HeaderView hview(buf_);
  return hview.size();
}

/**
 * Sets the stream selection table (map of trigger selections for each
 * storage manager output stream) in the response.
 */
void ConsRegResponseBuilder::
setStreamSelectionTable(std::map<std::string, Strings> const& selTable)
{
  // add the table just beyond the existing data
  uint8* bufPtr = buf_ + size();

  // add the number of entries in the table to the message
  convert (static_cast<uint32>(selTable.size()), bufPtr);
  bufPtr += sizeof(uint32);
  assert(((uint32) (bufPtr - buf_)) <= bufSize_);

  // add each entry in the table to the message
  std::map<std::string, Strings>::const_iterator mapIter;
  for (mapIter = selTable.begin(); mapIter != selTable.end(); mapIter++)
    {
      // create a new string list with the map key as the last entry
      Strings workList = mapIter->second;
      workList.push_back(mapIter->first);

      // copy the string list into the message
      bufPtr = MsgTools::fillNames(workList, bufPtr);
      assert(((uint32) (bufPtr - buf_)) <= bufSize_);
    }

  // update the message header with the new full size
  new (buf_) Header(Header::CONS_REG_RESPONSE, (bufPtr - buf_));
}

/**
 * Constructor for the consumer registration response viewer.
 */
ConsRegResponseView::ConsRegResponseView(void* buf):
  buf_((uint8*)buf),head_(buf)
{
  // verify that the buffer actually contains a registration response
  if (this->code() != Header::CONS_REG_RESPONSE)
    {
      throw cms::Exception("MessageDecoding","ConsRegResponseView")
        << "Invalid consumer registration response message code ("
        << this->code() << "). Should be " << Header::CONS_REG_RESPONSE << "\n";
    }

  // update the buffer pointer to just beyond the header
  uint8* bufPtr = buf_ + sizeof(Header);

  // decode the status
  status_ = convert32(bufPtr);
  bufPtr += sizeof(uint32);

  // decode the consumer ID
  consumerId_ = convert32(bufPtr);
  bufPtr += sizeof(uint32);

  assert(bufPtr); // silence clang static analyzer
}

/**
 * Returns the map of trigger selections for each storage manager
 * output stream.
 */
std::map<std::string, Strings> ConsRegResponseView::getStreamSelectionTable()
{
  std::map<std::string, Strings> selTable;

  // check if there is more than just the status code and consumer id
  if (size() >= (3 * sizeof(uint32)))
    {
      // initialize the data pointer to the start of the map data
      uint8* bufPtr = buf_ + sizeof(Header);
      bufPtr += (2 * sizeof(uint32));

      // decode the number of streams in the table
      uint32 streamCount = convert32(bufPtr);
      bufPtr += sizeof(uint32);

      // loop over each stream
      for (uint32 idx = 0; idx < streamCount; idx++)
        {
          // decode the vector of strings for the stream
          Strings workList;
          //uint32 listCount = convert32(bufPtr);
          bufPtr += sizeof(uint32);
          uint32 listLen = convert32(bufPtr);
          bufPtr += sizeof(uint32);
          MsgTools::getNames(bufPtr, listLen, workList);

          // pull the map key off the end of the list 
          std::string streamLabel = workList.back();
          workList.pop_back();
          selTable[streamLabel] = workList;

          // move on to the next entry in the message
          bufPtr += listLen;
        }
    }

  return selTable;
}
