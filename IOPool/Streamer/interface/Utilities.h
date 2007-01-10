#ifndef Streamer_Utilties_h
#define Streamer_Utilties_h

// -*- C++ -*-
//
//  This header presents several classes used internally by the
//  Streamer package.
//

#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "IOPool/Streamer/interface/Messages.h"
#include "IOPool/Streamer/interface/EventBuffer.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

#include "TBuffer.h"
#include "TClass.h"

#include <memory>
#include <string>
#include <fstream>
#include <vector>
#include <utility>
#include <iostream>

namespace edm
{
  edm::ProductRegistry getRegFromFile(const std::string& filename);
  std::auto_ptr<SendJobHeader> readHeaderFromStream(std::ifstream& ist);
  bool registryIsSubset(const SendJobHeader&, const ProductRegistry& reg);
  bool registryIsSubset(const SendJobHeader& sd, const SendJobHeader& ref);
  //bool registryIsSame(const SendJobHeader& sd, const SendJobHeader& ref);
  void mergeWithRegistry(const SendDescs& descs, ProductRegistry& reg);
  void declareStreamers(const SendDescs& descs);
  void buildClassCache(const SendDescs& descs);

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

#if 0
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
#endif

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

#if 0
  class EventInserter
  {
  public:
    explicit EventInserter(EventBuffer& b):buf_(&b) { }
    void insert(const EventMsg& msg,const ProductRegistry& prods)
    {
      std::auto_ptr<EventPrincipal> p = decoder_.decodeEvent(msg,prods);
	  send(p);
    }

	std::auto_ptr<EventPrincipal> decode(const EventMsg& msg,
	                                     const ProductRegistry& prods)
	{
      return decoder_.decodeEvent(msg,prods);
	}

	void send(std::auto_ptr<EventPrincipal> p)
	{
      EventBuffer::ProducerBuffer b(*buf_);
      void** v = (void**)b.buffer();
	  FDEBUG(2) << "Insert: event ptr = " << (void*)p.get() << std::endl;
      *v = p.release();
	  FDEBUG(2) << "Insert: " << b.buffer() << " " << b.size() << std::endl;
      b.commit(sizeof(void*));
	}

  private:
    EventBuffer* buf_;
    EventDecoder decoder_;
  };
#endif

  class EventExtractor
  {
  public:
    explicit EventExtractor(EventBuffer& b):buf_(&b) { }
    std::auto_ptr<EventPrincipal> extract()
    {
      EventBuffer::ConsumerBuffer b(*buf_);
	  FDEBUG(2) << "Extract: " << b.buffer() << " " << b.size() << std::endl;
      std::auto_ptr<EventPrincipal> p(*(EventPrincipal**)b.buffer());
	  FDEBUG(2) << "Extract: event ptr = " << (void*)p.get() << std::endl;
      return p;
    }
  private:
    EventBuffer* buf_;
  };

#if 0
  class EventReader
  {
  public:
    typedef std::vector<char> Buf;

    EventReader(std::ifstream& ist):
      ist_(&ist),b_(1000*1000*7) { }
    std::auto_ptr<EventPrincipal> read(const ProductRegistry& prods);
    int readMessage(Buf& here);

  private:

    std::ifstream* ist_;
    Buf b_;
    EventDecoder decoder_;
  };
#endif

}

#endif

