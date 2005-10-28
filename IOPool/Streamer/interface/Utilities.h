#ifndef IOP_STR_UTIL_HPP
#define IOP_STR_UTIL_HPP

#include "IOPool/StreamerData/interface/StreamedProducts.h"
#include "IOPool/StreamerData/interface/Messages.h"
#include "IOPool/Streamer/interface/EventBuffer.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/ProductRegistry.h"

#include "TBuffer.h"
#include "TClass.h"

#include <memory>

namespace edm
{
  bool registryIsSubset(const SendJobHeader&, const ProductRegistry&);
  void mergeWithRegistry(const SendJobHeader&, ProductRegistry&);

  class JobHeaderDecoder
  {
  public:
    JobHeaderDecoder();
    ~JobHeaderDecoder();

    std::auto_ptr<SendJobHeader> decodeJobHeader(const InitMsg& msg);

  private:
    TClass* desc_;
    TBuffer buf_;
  };

  class EventDecoder
  {
  public:
    EventDecoder();
    ~EventDecoder();

    std::auto_ptr<EventPrincipal> decodeEvent(const EventMsg&, 
					      const ProductRegistry&);

  private:
    //std::auto_ptr<SendEvent> decodeMsg(const EventMsg& msg);

    TClass* desc_;
    TBuffer buf_;
  };

  // ------------------------------------------------------------
  // Below are a few utilities for putting event pointers and job
  // header pointers onto the queue and extract them.

  class JobHeaderInserter
  {
  public:
    explicit JobHeaderInserter(EventBuffer& b):buf_(&b) { }
    void insert(const InitMsg& msg)
    {
      std::auto_ptr<SendJobHeader> p = decoder_.decodeJobHeader(msg);
      EventBuffer::ProducerBuffer b(*buf_);
      void** v = (void**)b.buffer();
      *v = p.release();
      b.commit(sizeof(void*));
    }
  private:
    EventBuffer* buf_;
    JobHeaderDecoder decoder_;
  };

  class JobHeaderExtractor
  {
  public:
    explicit JobHeaderExtractor(EventBuffer& b):buf_(&b) { }
    std::auto_ptr<SendJobHeader> extract()
    {
      EventBuffer::ConsumerBuffer b(*buf_);
      std::auto_ptr<SendJobHeader> p((SendJobHeader*)b.buffer());
      return p;
    }
  private:
    EventBuffer* buf_;
  };

  // -----------------

  class EventInserter
  {
  public:
    explicit EventInserter(EventBuffer& b):buf_(&b) { }
    void insert(const EventMsg& msg,const ProductRegistry& prods)
    {
      std::auto_ptr<EventPrincipal> p = decoder_.decodeEvent(msg,prods);
      EventBuffer::ProducerBuffer b(*buf_);
      void** v = (void**)b.buffer();
      *v = p.release();
      b.commit(sizeof(void*));
    }
  private:
    EventBuffer* buf_;
    EventDecoder decoder_;
  };

  class EventExtractor
  {
  public:
    explicit EventExtractor(EventBuffer& b):buf_(&b) { }
    std::auto_ptr<EventPrincipal> extract()
    {
      EventBuffer::ConsumerBuffer b(*buf_);
      std::auto_ptr<EventPrincipal> p((EventPrincipal*)b.buffer());
      return p;
    }
  private:
    EventBuffer* buf_;
  };

}

#endif

