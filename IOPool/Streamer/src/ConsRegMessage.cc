/**
 * These classes are used to build and view
 * the registration requests and replies that are exchanged between
 * individual event consumers and the event server.
 *
 * 15-Aug-2006 - KAB  - Initial Implementation
 */

#include "IOPool/Streamer/interface/ConsRegMessage.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

/**
 * Constructor for the consumer registration request builder.
 */
ConsRegRequestBuilder::ConsRegRequestBuilder(void* buf, uint32 bufSize,
                                             std::string const& consumerName,
                                             std::string const& consumerPriority,
                                             std::string const& requestParamSet):
  buf_((uint8*)buf),bufSize_(bufSize)
{
  uint8* bufPtr;
  uint32 len;

  // update the buffer pointer to just beyond the header
  bufPtr = buf_ + sizeof(Header);
  //cout << "bufPtr = 0x" << hex << ((uint32) bufPtr) << dec << endl;
  //cout << "buf_ = 0x" << hex << ((uint32) buf_) << dec << endl;
  //cout << "bufSize_ = " << bufSize_ << endl;
  assert(((uint32) (bufPtr - buf_)) <= bufSize_);

  // copy the consumer name into the message
  len = consumerName.length();
  assert(((uint32) (bufPtr + len + sizeof(uint32) - buf_)) <= bufSize_);
  convert(len, bufPtr);
  bufPtr += sizeof(uint32);
  consumerName.copy((char *) bufPtr, len);
  bufPtr += len;

  // copy the consumer priority into the message
  len = consumerPriority.length();
  assert(((uint32) (bufPtr + len + sizeof(uint32) - buf_)) <= bufSize_);
  convert(len, bufPtr);
  bufPtr += sizeof(uint32);
  consumerPriority.copy((char *) bufPtr, len);
  bufPtr += len;

  // copy the request parameter set into the message
  len = requestParamSet.length();
  assert(((uint32) (bufPtr + len + sizeof(uint32) - buf_)) <= bufSize_);
  convert(len, bufPtr);
  bufPtr += sizeof(uint32);
  requestParamSet.copy((char *) bufPtr, len);
  bufPtr += len;

  // create the message header now that we now the full size
  //cout << "bufPtr = 0x" << hex << ((uint32) bufPtr) << dec << endl;
  //cout << "buf_ = 0x" << hex << ((uint32) buf_) << dec << endl;
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
  uint8* bufPtr;
  uint32 len;

  // verify that the buffer actually contains a registration request
  if (this->code() != Header::CONS_REG_REQUEST)
    {
      throw cms::Exception("MessageDecoding","ConsRegRequestView")
        << "Invalid consumer registration request message code ("
        << this->code() << "). Should be " << Header::CONS_REG_REQUEST << "\n";
    }

  // update the buffer pointer to just beyond the header
  bufPtr = buf_ + sizeof(Header);

  // determine the consumer name
  len = convert32(bufPtr);
  bufPtr += sizeof(uint32);
  if (len >= 0)
    {
      if (len <= 256)
        {
          consumerName_.append((char *) bufPtr, len);
        }
      bufPtr += len;
    }

  // determine the consumer priority
  len = convert32(bufPtr);
  bufPtr += sizeof(uint32);
  if (len >= 0)
    {
      if (len <= 64)
        {
          consumerPriority_.append((char *) bufPtr, len);
        }
      bufPtr += len;
    }

  // determine the request parameter set (maintain backward compatibility
  // with sources of registration requests that don't have the param set)
  if (bufPtr < (buf_ + this->size()))
    {
      len = convert32(bufPtr);
      bufPtr += sizeof(uint32);
      if (len >= 0)
        {
          // what is a reasonable limit?  This is just to prevent
          // a bogus, really large value from being used...
          if (len <= 65000)
            {
              requestParameterSet_.append((char *) bufPtr, len);
            }
          bufPtr += len;
        }
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
  uint8* bufPtr;

  // update the buffer pointer to just beyond the header
  bufPtr = buf_ + sizeof(Header);
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
 * Constructor for the consumer registration response viewer.
 */
ConsRegResponseView::ConsRegResponseView(void* buf):
  buf_((uint8*)buf),head_(buf)
{
  uint8* bufPtr;

  // verify that the buffer actually contains a registration response
  if (this->code() != Header::CONS_REG_RESPONSE)
    {
      throw cms::Exception("MessageDecoding","ConsRegResponseView")
        << "Invalid consumer registration response message code ("
        << this->code() << "). Should be " << Header::CONS_REG_RESPONSE << "\n";
    }

  // update the buffer pointer to just beyond the header
  bufPtr = buf_ + sizeof(Header);

  // decode the status
  status_ = convert32(bufPtr);
  bufPtr += sizeof(uint32);

  // decode the consumer ID
  consumerId_ = convert32(bufPtr);
  bufPtr += sizeof(uint32);
}
