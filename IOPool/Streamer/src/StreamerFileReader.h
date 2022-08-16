#ifndef IOPool_Streamer_StreamerFileReader_h
#define IOPool_Streamer_StreamerFileReader_h

#include "IOPool/Streamer/interface/StreamerInputSource.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

#include <memory>
#include <string>
#include <vector>

class EventMsgView;
class InitMsgView;

namespace edm {
  class ConfigurationDescriptions;
  class EventPrincipal;
  class EventSkipperByID;
  class FileCatalogItem;
  struct InputSourceDescription;
  class ParameterSet;
  class StreamerInputFile;
  class StreamerFileReader : public StreamerInputSource {
  public:
    StreamerFileReader(ParameterSet const& pset, InputSourceDescription const& desc);
    ~StreamerFileReader() override;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    InitMsgView const* getHeader();
    EventMsgView const* getNextEvent();
    bool newHeader();

    Next checkNext() override;
    void skip(int toSkip) override;
    void genuineReadFile() override;
    void genuineCloseFile() override;
    void reset_() override;

    std::shared_ptr<EventSkipperByID const> eventSkipperByID() const { return get_underlying_safe(eventSkipperByID_); }
    std::shared_ptr<EventSkipperByID>& eventSkipperByID() { return get_underlying_safe(eventSkipperByID_); }

    std::vector<FileCatalogItem> streamerNames_;  // names of Streamer files
    edm::propagate_const<std::unique_ptr<StreamerInputFile>> streamReader_;
    edm::propagate_const<std::shared_ptr<EventSkipperByID>> eventSkipperByID_;
    int initialNumberOfEventsToSkip_;
    int prefetchMBytes_;
    bool isFirstFile_ = true;
  };
}  // namespace edm

#endif
