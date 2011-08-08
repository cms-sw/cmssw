#ifndef IOPool_Streamer_StreamerFileReader_h
#define IOPool_Streamer_StreamerFileReader_h

#include "IOPool/Streamer/interface/StreamerInputSource.h"

#include "boost/shared_ptr.hpp"

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
    virtual EventPrincipal* read();

    InitMsgView const* getHeader(); 
    EventMsgView const* getNextEvent();
    bool const newHeader(); 
    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:  
    std::vector<std::string> streamerNames_; // names of Streamer files
    std::auto_ptr<StreamerInputFile> streamReader_;
    boost::shared_ptr<EventSkipperByID> eventSkipperByID_;
    int numberOfEventsToSkip_;
  };
} //end-of-namespace-def

#endif

