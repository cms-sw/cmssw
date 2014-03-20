#ifndef IOPool_Streamer_DQMStreamerReader_h
#define IOPool_Streamer_DQMStreamerReader_h

#include "IOPool/Streamer/interface/StreamerInputSource.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "boost/shared_ptr.hpp"
#include "boost/filesystem.hpp"

#include <memory>
#include <string>
#include <vector>
#include <iterator>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

class EventMsgView;
class InitMsgView;

namespace edm {
  class ConfigurationDescriptions;
  class EventPrincipal;
  class EventSkipperByID;
  struct InputSourceDescription;
  class ParameterSet;
  class StreamerInputFile;
  class DQMStreamerReader : public StreamerInputSource {
  public:
    DQMStreamerReader(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~DQMStreamerReader();

    InitMsgView const* getHeader();
    EventMsgView const* getNextEvent();
    bool newHeader();
    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    virtual bool checkNextEvent();
    virtual void skip(int toSkip);
    virtual void closeFile_();
    virtual void reset_();
    
    virtual bool checkNewData(int lumi);
    virtual bool checkNextLS();
    virtual void openNewFile(std::string filename);
    std::string getDataFile(int lumi);
    //
    unsigned int runNumber_;
    std::string dqmInputDir_;
    unsigned int currentLumiSection_;
    unsigned int totalEventPerLs_;
    unsigned int processedEventPerLs_;
    //
    typedef unsigned int run_t;
    typedef unsigned int lumisection_t;
    static const std::size_t run_min_length = 6;
    static const std::size_t lumisection_min_length = 4;
    static std::string make_path(run_t run, lumisection_t lumisection);
    static std::string to_padded_string(int n, std::size_t min_length);

    struct DQMJSON
    {
      std::size_t n_events;
      std::string datafilename;
      std::string definition;
      std::string source;
      
      void load(run_t run, lumisection_t lumisection)
      {
        boost::property_tree::ptree pt;
        read_json(make_path(run, lumisection), pt);
	
        // We rely on n_events to be the first item on the array...
        n_events = pt.get_child("data").front().second.get_value<std::size_t>();
        datafilename = std::next(pt.get_child("data").begin(),2)->second.get_value<std::string>();
        definition = pt.get<std::string>("definition");
        source = pt.get<std::string>("source");
      }
    };

    std::string  streamerName_;
    std::unique_ptr<StreamerInputFile> streamReader_;
    boost::shared_ptr<EventSkipperByID> eventSkipperByID_;
    int initialNumberOfEventsToSkip_;
  };
} //end-of-namespace-def

#endif
