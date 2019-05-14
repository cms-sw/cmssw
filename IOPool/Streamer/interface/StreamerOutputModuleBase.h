#ifndef IOPool_Streamer_StreamerOutputModuleBase_h
#define IOPool_Streamer_StreamerOutputModuleBase_h

#include "IOPool/Streamer/interface/StreamerOutputModuleCommon.h"
#include "FWCore/Framework/interface/one/OutputModule.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "IOPool/Streamer/interface/MsgTools.h"
//#include "IOPool/Streamer/interface/StreamSerializer.h"
//#include <memory>
//#include <vector>

class InitMsgBuilder;
class EventMsgBuilder;
namespace edm {
  class ParameterSetDescription;

  typedef detail::TriggerResultsBasedEventSelector::handle_t Trig;

  class StreamerOutputModuleBase : public one::OutputModule<one::WatchRuns, one::WatchLuminosityBlocks>,
                                   StreamerOutputModuleCommon {
  public:
    explicit StreamerOutputModuleBase(ParameterSet const& ps);
    ~StreamerOutputModuleBase() override;
    static void fillDescription(ParameterSetDescription& desc);

  private:
    void beginRun(RunForOutput const&) override;
    void endRun(RunForOutput const&) override;
    void beginJob() override;
    void endJob() override;
    void writeRun(RunForOutput const&) override;
    void writeLuminosityBlock(LuminosityBlockForOutput const&) override;
    void write(EventForOutput const& e) override;

    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void doOutputHeader(InitMsgBuilder const& init_message) = 0;
    virtual void doOutputEvent(EventMsgBuilder const& msg) = 0;

    Trig getTriggerResults(EDGetTokenT<TriggerResults> const& token, EventForOutput const& e) const;

  private:
    edm::EDGetTokenT<edm::TriggerResults> trToken_;

  };  //end-of-class-def

}  // namespace edm

#endif
