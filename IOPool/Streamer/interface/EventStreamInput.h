#ifndef Streamer_EventStreamInput_h
#define Streamer_EventStreamInput_h

/*----------------------------------------------------------------------

Event streaming input service

$Id: EventStreamInput.h,v 1.2 2005/08/25 05:17:22 jbk Exp $

----------------------------------------------------------------------*/

#include <vector>
#include <memory>
#include <string>
#include <fstream>

#include "IOPool/Streamer/interface/EventBuffer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/InputService.h"
#include "FWCore/Framework/interface/Retriever.h"
#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/BranchKey.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventProvenance.h"
#include "FWCore/Framework/interface/EventAux.h"

class TClass;

namespace edm {

  class EventStreamerInputImpl
  {
  public:
    //------------------------------------------------------------
    // Nested class PoolRetriever: pretends to support file reading.
    //
    class StreamRetriever : public Retriever
    {
    public:
      virtual ~StreamRetriever();
      virtual std::auto_ptr<EDProduct> get(BranchKey const& k);
    };

    //------------------------------------------------------------

    EventStreamerInputImpl(ParameterSet const& pset,
		     InputServiceDescription const& desc,
		     EventBuffer* bufs);
    ~EventStreamerInputImpl();

    std::auto_ptr<EventPrincipal> reconstitute();
    
    std::vector<char>& registryBuffer() { return regbuf_; }
    void decodeRegistry();
  private:
    std::vector<char> regbuf_;
    EventBuffer* bufs_;
    ProductRegistry* pr_;
    TClass* send_event_;
    StreamRetriever store_;

    void init();

    // EventAux not handled
  };

  template <class Producer>
  class EventStreamInput : public InputService
  {
  public:
    EventStreamInput(ParameterSet const& pset,
		     InputServiceDescription const& desc);
    virtual ~EventStreamInput();

    virtual std::auto_ptr<EventPrincipal> read();
  private:
    EventBuffer* bufs_;
    EventStreamerInputImpl es_;
    Producer p_;
  };

  // --------------------------------

  template <class Producer>
  EventStreamInput<Producer>::EventStreamInput(ParameterSet const& ps,
					       InputServiceDescription const& reg):
    InputService(reg),
    bufs_(getEventBuffer(ps.template getParameter<int>("max_event_size"),
			 ps.template getParameter<int>("max_queue_depth"))),
    es_(ps,reg,bufs_),
    p_(ps.template getParameter<ParameterSet>("producer_config"),*reg.preg_,bufs_)
  {
    p_.getRegistry(es_.registryBuffer());
    es_.decodeRegistry();
  }

  template <class Producer>
  EventStreamInput<Producer>::~EventStreamInput()
  {
    try {
      p_.stop(); // should not throw !
    }
    catch(...)
      {
	std::cerr << "EventStreamInput: stopping the producer caused "
		  << "an exception!\n"
		  << "Igoring the exception." << std::endl;
      }
  }

  template <class Producer>
  std::auto_ptr<EventPrincipal> EventStreamInput<Producer>::read()
  {
    p_.needBuffer();
    return es_.reconstitute();
  }
  
}
#endif
