#ifndef Streamer_EventStreamOutput_h
#define Streamer_EventStreamOutput_h

// -*- C++ -*-

/*
  If the consumer is multithreaded and the destructor of the
  consumer does not return until the thread behind the scenes is
  stopped and dead, then we do not need to do anything special with
  the EventBuffer data member.  The current code does not make this
  assumption (although it probably should) and keeps the EventBuffer
  objects around until global destructors are run (after main).

  The EventBuffer in the output module defined below is shared between
  the consumer and the impl.  The protocol here is that the output module
  is given an event to serialize into the EventBuffer object.  The
  Consumer is poked to notify it that an event has been added to the
  EventBuffer.  Of course the poking is not necessary if the consumer
  has started a thread that is blocked on data arriving on the queue
  in the EventBuffer.
 */

#include "IOPool/Streamer/interface/BufferArea.h"
#include "IOPool/Streamer/interface/EventBuffer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/OutputModule.h"

#include <iostream>
#include <vector>

class TClass;

namespace edm
{
  // this impl class prevents several unpleasant things from appearing in
  // a header file

  class EventStreamerImpl
  {
  public:
    typedef std::vector<char> ProdRegBuf;
    typedef OutputModule::Selections Selections;

    EventStreamerImpl(ParameterSet const& ps,
		      Selections const* selections,
		      EventBuffer* bufs);
    ~EventStreamerImpl();

    void serialize(EventPrincipal const& e);
    void serializeRegistry(Selections const& prods);

    void* registryBuffer() const { return (void*)&prod_reg_buf_[0]; }
    int registryBufferSize() const { return prod_reg_len_; }

  private:

    Selections const* selections_;
    EventBuffer* bufs_;
    TClass* tc_; // for SendEvent
    ProdRegBuf prod_reg_buf_;
    int prod_reg_len_;
  };

  // ----------------------------------------------------------

  template <class Consumer>
  class EventStreamingModule : public OutputModule
  {
  public:
    explicit EventStreamingModule(ParameterSet const& ps);

    virtual ~EventStreamingModule();
  private:
    virtual void write(EventPrincipal const& e);
    virtual void beginJob(EventSetup const&);
    virtual void endLuminosityBlock(LuminosityBlockPrincipal const&){}
    virtual void endRun(RunPrincipal const&){}
    // bufs_ needs to live until the end of all threads using it (end of job)
    EventBuffer* bufs_;
    EventStreamerImpl es_;
    Consumer c_;
  };

  // ----------------------------------------------------------

  /*
    Requires Consumer to provide:
      ctor(ParameterSe const&, ProductRegistry const&, EventBuffer*)
      void bufferReady();
      void stop();
      void sendRegistry(void* buffer, int len);
   */

  template <class Consumer>
  EventStreamingModule<Consumer>::EventStreamingModule(ParameterSet const& ps):
    OutputModule(ps),
    bufs_(getEventBuffer(ps.template getParameter<int>("max_event_size"),
			 ps.template getParameter<int>("max_queue_depth"))),
    es_(ps,&descVec()[InEvent],bufs_),
    c_(ps.template getParameter<ParameterSet>("consumer_config"),bufs_)
  {
    // temporary hack
    //EventSetup* setu=0;
    //this->beginJob(*setu);
  }

  template <class Consumer>
  EventStreamingModule<Consumer>::~EventStreamingModule()
  {
    try {
      c_.stop(); // should not throw !
    }
    catch(...)
      {
	std::cerr << "EventStreamingModule: stopping the consumer caused "
		  << "an exception!\n"
		  << "Igoring the exception." << std::endl;
      }
  }

  template <class Consumer>
  void EventStreamingModule<Consumer>::write(EventPrincipal const& e)
  {
    // b contains the serialized data, the provenance information, and event
    // header data - the collision ID and trigger bits, the object lifetime
    // is managed by the es_ object.
    es_.serialize(e);
    c_.bufferReady();
  }

  template <class Consumer>
  void EventStreamingModule<Consumer>::beginJob(EventSetup const&)
  {
    //std::cerr << "In beginJob" << std::endl;

    es_.serializeRegistry(descVec()[InEvent]);
    c_.sendRegistry(es_.registryBuffer(),es_.registryBufferSize());

  }


}

#endif
