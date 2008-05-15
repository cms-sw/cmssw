#ifndef IOPool_Streamer_StreamerOutputModule_h
#define IOPool_Streamer_StreamerOutputModule_h

// $Id: StreamerOutputModule.h,v 1.38 2008/04/18 20:21:12 biery Exp $

#include "IOPool/Streamer/interface/StreamerOutputModuleBase.h"

namespace edm {
  template <class Consumer>
  class StreamerOutputModule : public StreamerOutputModuleBase {

  /** Consumers are suppose to provide
         void doOutputHeader(InitMsgBuilder const& init_message)
         void doOutputEvent(EventMsgBuilder const& msg)
         void start()
         void stop()
  **/
        
  public:
    explicit StreamerOutputModule(ParameterSet const& ps);  
    virtual ~StreamerOutputModule();

  private:
    virtual void start() const;
    virtual void stop() const;
    virtual void doOutputHeader(InitMsgBuilder const& init_message) const;
    virtual void doOutputEvent(EventMsgBuilder const& msg) const;

  private:
    std::auto_ptr<Consumer> c_;
  }; //end-of-class-def

 

  template <class Consumer>
  StreamerOutputModule<Consumer>::StreamerOutputModule(ParameterSet const& ps) :
    StreamerOutputModuleBase(ps),
    c_(new Consumer(ps)) {
  }

  template <class Consumer>
  StreamerOutputModule<Consumer>::~StreamerOutputModule() {}

  template <class Consumer>
  void
  StreamerOutputModule<Consumer>::start() const {
    c_->start();
  }
  
  template <class Consumer>
  void
  StreamerOutputModule<Consumer>::stop() const {
    c_->stop();
  }

  template <class Consumer>
  void
  StreamerOutputModule<Consumer>::doOutputHeader(InitMsgBuilder const& init_message) const {
    c_->doOutputHeader(init_message);
  }
   
//______________________________________________________________________________
  template <class Consumer>
  void
  StreamerOutputModule<Consumer>::doOutputEvent(EventMsgBuilder const& msg) const {
    c_->doOutputEvent(msg); // You can't use msg in StreamerOutputModule after this point
  }
} // end of namespace-edm

#endif

