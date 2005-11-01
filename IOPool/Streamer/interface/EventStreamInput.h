#ifndef Streamer_EventStreamInput_h
#define Streamer_EventStreamInput_h

#error "this way has been changed to use the functions in Utilities.h"

/*----------------------------------------------------------------------

Event streaming input source

$Id: EventStreamInput.h,v 1.5 2005/10/28 04:39:31 jbk Exp $

----------------------------------------------------------------------*/

#include <vector>
#include <memory>
#include <string>
#include <fstream>

#include "IOPool/StreamerData/interface/Messages.h"
#include "IOPool/Streamer/interface/EventBuffer.h"
#include "IOPool/Streamer/interface/Utilities.h"

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
    TClass* prog_reg_;
    enum State { Init=0, Running=1 };
    State state_;

    void init();
    std::auto_ptr<EventPrincipal> doOneEvent(const EventMsg& msg);
    void decodeRegistry(const InitMsg& msg);


    // EventAux not handled
  };

  //------------------------------------------------------------
  /*
    Here is the actual input source template.
    
   */


  template <class Producer>
  class EventStreamInput : public InputSource
  {
  public:
    EventStreamInput(ParameterSet const& pset,
		     InputSourceDescription const& desc);
    virtual ~EventStreamInput();

    virtual std::auto_ptr<EventPrincipal> read();
  private:
    Producer p_;
    EventStreamerInputImpl es_;
    EventBuffer* bufs_;
  };

  // --------------------------------

  // JBK - I'll currently working on getting rid of this internal
  // class altogether because of the new utilities.  Also, the
  // JobHeaderDecoder/EventDecoder objects need to be added.
  // But where should the conversion from message to Send* object
  // take place?  In here or in the user code (template parameter)?
  // or split it?

  template <class Producer>
  EventStreamInput<Producer>::EventStreamInput(ParameterSet const& ps,
					       InputSourceDescription const& reg):
    InputSource(reg),
    p_(ps.template getParameter<ParameterSet>("producer_config"),*reg.preg_),
    es_(ps,reg,p_.getQueue()),
    bufs_(p_.getQueue())
  {
    // this next call may block waiting for a registry to arrive
    // the function getRegistry is expected to copy the registry message
    // into the buffer that is given as an out argument
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
    // the next call may do nothing if the buffer comes from another thread
    // one of the next calls is going to block waiting for I/O
    p_.needBuffer();
    return es_.reconstitute();
  }
  
}
#endif
