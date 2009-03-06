#ifndef IOPool_Streamer_StreamerOutputModuleBase_h
#define IOPool_Streamer_StreamerOutputModuleBase_h

#include "FWCore/Framework/interface/OutputModule.h"
#include "IOPool/Streamer/interface/StreamSerializer.h"
#include "IOPool/Streamer/interface/MsgTools.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <memory>
#include <vector>

class InitMsgBuilder;
class EventMsgBuilder;
namespace edm {
  class StreamerOutputModuleBase : public OutputModule {
  public:
    explicit StreamerOutputModuleBase(ParameterSet const& ps);  
    virtual ~StreamerOutputModuleBase();

  private:
    virtual void beginRun(RunPrincipal const&);
    virtual void endRun(RunPrincipal const&);
    virtual void beginJob(EventSetup const&);
    virtual void endJob();
    virtual void writeRun(RunPrincipal const&);
    virtual void writeLuminosityBlock(LuminosityBlockPrincipal const&);
    virtual void write(EventPrincipal const& e);

    virtual void start() const = 0;
    virtual void stop() const = 0;
    virtual void doOutputHeader(InitMsgBuilder const& init_message) const = 0;
    virtual void doOutputEvent(EventMsgBuilder const& msg) const = 0;

    std::auto_ptr<InitMsgBuilder> serializeRegistry();
    std::auto_ptr<EventMsgBuilder> serializeEvent(EventPrincipal const& e); 
    void setHltMask(EventPrincipal const& e);
    void setLumiSection();

  private:
    Selections const* selections_;

    int maxEventSize_;
    bool useCompression_;
    int compressionLevel_;

    // test luminosity sections
    int lumiSectionInterval_;  
    double timeInSecSinceUTC;

    StreamSerializer serializer_;

    //Event variables, made class memebers to avoid re instatiation for each event.
    unsigned int hltsize_;
    uint32 lumi_;
    std::vector<bool> l1bit_;
    std::vector<unsigned char> hltbits_;
    uint32 origSize_;

    Strings hltTriggerSelections_;
    uint32 outputModuleId_;
  }; //end-of-class-def
} // end of namespace-edm

#endif
