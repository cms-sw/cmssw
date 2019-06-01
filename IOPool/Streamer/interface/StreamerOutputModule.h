#ifndef IOPool_Streamer_StreamerOutputModule_h
#define IOPool_Streamer_StreamerOutputModule_h

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "IOPool/Streamer/interface/StreamerOutputModuleBase.h"

namespace edm {
  template <typename Consumer>
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
    ~StreamerOutputModule() override;
    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    void start() override;
    void stop() override;
    void doOutputHeader(InitMsgBuilder const& init_message) override;
    void doOutputEvent(EventMsgBuilder const& msg) override;
    void beginLuminosityBlock(edm::LuminosityBlockForOutput const&) override;
    void endLuminosityBlock(edm::LuminosityBlockForOutput const&) override;

  private:
    edm::propagate_const<std::unique_ptr<Consumer>> c_;
  };  //end-of-class-def

  template <typename Consumer>
  StreamerOutputModule<Consumer>::StreamerOutputModule(ParameterSet const& ps)
      : edm::one::OutputModuleBase::OutputModuleBase(ps), StreamerOutputModuleBase(ps), c_(new Consumer(ps)) {}

  template <typename Consumer>
  StreamerOutputModule<Consumer>::~StreamerOutputModule() {}

  template <typename Consumer>
  void StreamerOutputModule<Consumer>::start() {
    c_->start();
  }

  template <typename Consumer>
  void StreamerOutputModule<Consumer>::stop() {
    c_->stop();
  }

  template <typename Consumer>
  void StreamerOutputModule<Consumer>::doOutputHeader(InitMsgBuilder const& init_message) {
    c_->doOutputHeader(init_message);
  }

  //______________________________________________________________________________
  template <typename Consumer>
  void StreamerOutputModule<Consumer>::doOutputEvent(EventMsgBuilder const& msg) {
    c_->doOutputEvent(msg);  // You can't use msg in StreamerOutputModule after this point
  }

  template <typename Consumer>
  void StreamerOutputModule<Consumer>::beginLuminosityBlock(edm::LuminosityBlockForOutput const&) {}

  template <typename Consumer>
  void StreamerOutputModule<Consumer>::endLuminosityBlock(edm::LuminosityBlockForOutput const&) {}

  template <typename Consumer>
  void StreamerOutputModule<Consumer>::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    StreamerOutputModuleBase::fillDescription(desc);
    Consumer::fillDescription(desc);
    descriptions.add("streamerOutput", desc);
  }
}  // namespace edm

#endif
