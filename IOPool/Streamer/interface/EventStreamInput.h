#ifndef Streamer_EventStreamInput_h
#define Streamer_EventStreamInput_h

/*----------------------------------------------------------------------

Event streaming input source

$Id: EventStreamInput.h,v 1.3 2005/09/01 01:05:14 wmtan Exp $

----------------------------------------------------------------------*/

#include <vector>
#include <memory>
#include <string>
#include <fstream>

#include "IOPool/Streamer/interface/EventBuffer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/DelayedReader.h"
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

    EventStreamerInputImpl(ParameterSet const& pset,
		     InputSourceDescription const& desc,
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

    void init();

    // EventAux not handled
  };

  template <class Producer>
  class EventStreamInput : public InputSource
  {
  public:
    EventStreamInput(ParameterSet const& pset,
		     InputSourceDescription const& desc);
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
					       InputSourceDescription const& reg):
    InputSource(reg),
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
