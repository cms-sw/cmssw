#ifndef IOPool_Streamer_StreamerOutputModule_h
#define IOPool_Streamer_StreamerOutputModule_h

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "IOPool/Streamer/interface/StreamerOutputModuleBase.h"

namespace edm {
  template<typename Consumer>
  class StreamerOutputModule : public StreamerOutputModuleBase {

  /** Consumers are suppose to provide
         void doOutputHeader(InitMsgBuilder const& init_message)
         void doOutputEvent(EventMsgBuilder const& msg)
         void start()
         void stop()
	 static void fillDescription(ParameterSetDescription&)
  **/
        
  public:
    explicit StreamerOutputModule(ParameterSet const& ps);  
    virtual ~StreamerOutputModule();
    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    virtual void start() const;
    virtual void stop() const;
    virtual void doOutputHeader(InitMsgBuilder const& init_message) const;
    virtual void doOutputEvent(EventMsgBuilder const& msg) const;
    virtual void beginLuminosityBlock(edm::LuminosityBlockPrincipal const&, edm::ModuleCallingContext const*) override;
    virtual void endLuminosityBlock(edm::LuminosityBlockPrincipal const&, edm::ModuleCallingContext const*) override;

  private:
    std::auto_ptr<Consumer> c_;
  }; //end-of-class-def

  template<typename Consumer>
  StreamerOutputModule<Consumer>::StreamerOutputModule(ParameterSet const& ps) :
    edm::one::OutputModuleBase::OutputModuleBase(ps),
    StreamerOutputModuleBase(ps),
    c_(new Consumer(ps))
    {
  }

  template<typename Consumer>
  StreamerOutputModule<Consumer>::~StreamerOutputModule() {}

  template<typename Consumer>
  void
  StreamerOutputModule<Consumer>::start() const {
    c_->start();
  }
  
  template<typename Consumer>
  void
  StreamerOutputModule<Consumer>::stop() const {
    c_->stop();
  }

  template<typename Consumer>
  void
  StreamerOutputModule<Consumer>::doOutputHeader(InitMsgBuilder const& init_message) const {
    c_->doOutputHeader(init_message);
  }
   
//______________________________________________________________________________
  template<typename Consumer>
  void
  StreamerOutputModule<Consumer>::doOutputEvent(EventMsgBuilder const& msg) const {
    c_->doOutputEvent(msg); // You can't use msg in StreamerOutputModule after this point
  }

  template<typename Consumer>
  void
  StreamerOutputModule<Consumer>::beginLuminosityBlock(edm::LuminosityBlockPrincipal const&, edm::ModuleCallingContext const*) {}

  template<typename Consumer>
  void
  StreamerOutputModule<Consumer>::endLuminosityBlock(edm::LuminosityBlockPrincipal const&, edm::ModuleCallingContext const*) {}

  template<typename Consumer>
  void
  StreamerOutputModule<Consumer>::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    StreamerOutputModuleBase::fillDescription(desc);
    Consumer::fillDescription(desc);
    descriptions.add("streamerOutput", desc);
  }
} // end of namespace-edm

#endif
