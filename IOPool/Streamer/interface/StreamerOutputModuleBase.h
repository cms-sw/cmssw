#ifndef IOPool_Streamer_StreamerOutputModuleBase_h
#define IOPool_Streamer_StreamerOutputModuleBase_h

#include "FWCore/Framework/interface/one/OutputModule.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/StreamSerializer.h"
#include <memory>
#include <vector>

class InitMsgBuilder;
class EventMsgBuilder;
namespace edm {
  class ParameterSetDescription;

  typedef detail::TriggerResultsBasedEventSelector::handle_t Trig;

  class StreamerOutputModuleBase : public one::OutputModule<one::WatchRuns, one::WatchLuminosityBlocks> {
  public:
    explicit StreamerOutputModuleBase(ParameterSet const& ps);
    ~StreamerOutputModuleBase() override;
    static void fillDescription(ParameterSetDescription & desc);

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

    std::unique_ptr<InitMsgBuilder> serializeRegistry();
    std::unique_ptr<EventMsgBuilder> serializeEvent(EventForOutput const& e); 
    Trig getTriggerResults(EDGetTokenT<TriggerResults> const& token, EventForOutput const& e) const;
    void setHltMask(EventForOutput const& e);
    void setLumiSection();

  private:
    SelectedProducts const* selections_;

    int maxEventSize_;
    bool useCompression_;
    int compressionLevel_;

    // test luminosity sections
    int lumiSectionInterval_;  
    double timeInSecSinceUTC;

    StreamSerializer serializer_;

    SerializeDataBuffer serializeDataBuffer_;

    //Event variables, made class memebers to avoid re instatiation for each event.
    unsigned int hltsize_;
    uint32 lumi_;
    std::vector<bool> l1bit_;
    std::vector<unsigned char> hltbits_;
    uint32 origSize_;
    char host_name_[255];

    edm::EDGetTokenT<edm::TriggerResults> trToken_;
    Strings hltTriggerSelections_;
    uint32 outputModuleId_;
  }; //end-of-class-def
} // end of namespace-edm

#endif
