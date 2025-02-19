#ifndef IOPool_Streamer_ConsumerRegMessage_h
#define IOPool_Streamer_ConsumerRegMessage_h

/**
 * These classes are used to build and view
 * the registration requests and replies that are exchanged between
 * individual event consumers and the event server.
 *
 * 15-Aug-2006 - KAB  - Initial Implementation
 */

#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/MsgHeader.h"
#include <map>

// --------------- registration request builder ----------------

class ConsRegRequestBuilder
{
 public:
  ConsRegRequestBuilder(void* buf, uint32 bufSize,
                        std::string const& consumerName,
                        std::string const& requestParameterSet);

  uint32 bufferSize() const { return bufSize_; }
  uint8* startAddress() { return buf_; }
  uint32 size() const;

 private:
  uint8* buf_;
  uint32 bufSize_;
};

// --------------- registration request viewer -----------------

class ConsRegRequestView
{
 public:
  ConsRegRequestView(void* buf);

  uint32 code() const { return head_.code(); }
  uint32 size() const { return head_.size(); }
  uint8* startAddress() { return buf_; }

  std::string getConsumerName() { return consumerName_; }
  std::string getRequestParameterSet() { return requestParameterSet_; }

 private:
  uint8* buf_;
  HeaderView head_;
  std::string consumerName_;
  std::string requestParameterSet_;
};

// -------------- registration response builder ----------------

class ConsRegResponseBuilder
{
 public:
  ConsRegResponseBuilder(void* buf, uint32 bufSize,
                         uint32 status, uint32 consumerId);

  uint32 bufferSize() const { return bufSize_; }
  uint8* startAddress() { return buf_; }
  uint32 size() const;

  void setStreamSelectionTable(std::map<std::string, Strings> const& selTable);

  enum STATUS_CODES { ES_NOT_READY = 0x10000 };

 private:
  uint8* buf_;
  uint32 bufSize_;
};

// -------------- registration response viewer -----------------

class ConsRegResponseView
{
 public:
  ConsRegResponseView(void* buf);

  uint32 code() const { return head_.code(); }
  uint32 size() const { return head_.size(); }
  uint8* startAddress() { return buf_; }

  uint32 getStatus() { return status_; }
  uint32 getConsumerId() { return consumerId_; }
  std::map<std::string, Strings> getStreamSelectionTable();

 private:
  uint8* buf_;
  HeaderView head_;
  uint32 status_;
  uint32 consumerId_;
};

#endif
