#ifndef IOPool_Streamer_Utilties_h
#define IOPool_Streamer_Utilties_h

// -*- C++ -*-
//
//  This header presents several classes used internally by the
//  Streamer package.
//

#include "TBufferFile.h"

#include "FWCore/Utilities/interface/DebugMacros.h"
#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "IOPool/Streamer/interface/EventBuffer.h"

#include <iostream>

namespace edm
{
  class InitMsg;
  edm::ProductRegistry getRegFromFile(std::string const& filename);
  std::auto_ptr<SendJobHeader> readHeaderFromStream(std::ifstream& ist);
  bool registryIsSubset(SendJobHeader const&, ProductRegistry const& reg);
  bool registryIsSubset(SendJobHeader const& sd, SendJobHeader const& ref);
  //bool registryIsSame(SendJobHeader const& sd, SendJobHeader const& ref);

  class JobHeaderDecoder
  {
  public:
    JobHeaderDecoder();
    ~JobHeaderDecoder();

    std::auto_ptr<SendJobHeader> decodeJobHeader(InitMsg const& msg);

  private:
    TClass* desc_;
    TBufferFile buf_;
  };

  // ------------------------------------------------------------
  // Below are a few utilities for putting event pointers and job
  // header pointers onto the queue and extract them.

  class JobHeaderInserter
  {
  public:
    explicit JobHeaderInserter(EventBuffer& b) : buf_(&b) {}
    void insert(InitMsg const& msg)
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

  class EventExtractor
  {
  public:
    explicit EventExtractor(EventBuffer& b):buf_(&b) { }
    EventPrincipal* extract()
    {
      EventBuffer::ConsumerBuffer b(*buf_);
	  FDEBUG(2) << "Extract: " << b.buffer() << " " << b.size() << std::endl;
      EventPrincipal* p(*(EventPrincipal**)b.buffer());
	  FDEBUG(2) << "Extract: event ptr = " << (void*)p << std::endl;
      return p;
    }
  private:
    EventBuffer* buf_;
  };

}

#endif

