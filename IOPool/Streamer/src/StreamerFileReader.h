#ifndef IOPool_Streamer_StreamerFileReader_h
#define IOPool_Streamer_StreamerFileReader_h

#include "IOPool/Streamer/interface/StreamerInputSource.h"

#include <memory>
#include <string>
#include <vector>

class EventMsgView;
class InitMsgView;

namespace edm {
  class ConfigurationDescriptions;
  class EventPrincipal;
  class EventSkipperByID;
  struct InputSourceDescription;
  class ParameterSet;
  class StreamerInputFile;
  class StreamerFileReader : public StreamerInputSource {
  public:
    StreamerFileReader(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~StreamerFileReader();

    InitMsgView const* getHeader();
    EventMsgView const* getNextEvent();
    bool newHeader();
    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    virtual bool checkNextEvent();
    virtual void skip(int toSkip);
    virtual void genuineCloseFile() override;
    virtual void reset_();

    std::vector<std::string> streamerNames_; // names of Streamer files
    std::unique_ptr<StreamerInputFile> streamReader_;
    std::shared_ptr<EventSkipperByID> eventSkipperByID_;
    int initialNumberOfEventsToSkip_;
  };
} //end-of-namespace-def

#endif

