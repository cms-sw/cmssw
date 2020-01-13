#ifndef DQMServices_StreamerIO_DQMStreamerReader_h
#define DQMServices_StreamerIO_DQMStreamerReader_h

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "IOPool/Streamer/interface/StreamerInputSource.h"
#include "IOPool/Streamer/interface/StreamerInputFile.h"
#include "IOPool/Streamer/interface/MsgTools.h"

#include "DQMFileIterator.h"
#include "DQMMonitoringService.h"
#include "TriggerSelector.h"

#include "boost/filesystem.hpp"

#include <memory>
#include <string>
#include <vector>
#include <iterator>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace dqmservices {

  class DQMStreamerReader : public edm::StreamerInputSource {
  public:
    DQMStreamerReader(edm::ParameterSet const& pset, edm::InputSourceDescription const& desc);
    ~DQMStreamerReader() override;

    bool newHeader();
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    typedef std::vector<std::string> Strings;

  protected:
    Next checkNext() override;      /* from raw input source */
    void skip(int toSkip) override; /* from raw input source */
    void genuineReadFile() override;
    void genuineCloseFile() override;

  private:
    // our own, but we do inherit reset(),
    // which will break things if called
    void reset_() override;

    void openFileImp_(const DQMFileIterator::LumiEntry& entry);
    void closeFileImp_(const std::string& reason);

    bool openNextFileImp_();

    InitMsgView const* getHeaderMsg();
    EventMsgView const* getEventMsg();

    EventMsgView const* prepareNextEvent();
    bool prepareNextFile();
    bool acceptEvent(const EventMsgView*);

    bool triggerSel();
    bool matchTriggerSel(Strings const& tnames);
    bool acceptAllEvt_;
    bool matchTriggerSel_;
    bool isFirstFile_ = true;

    unsigned int runNumber_;
    std::string runInputDir_;
    std::string streamLabel_;
    Strings hltSel_;

    unsigned int processedEventPerLs_;
    unsigned int minEventsPerLs_;

    bool flagSkipFirstLumis_;
    bool flagEndOfRunKills_;
    bool flagDeleteDatFiles_;

    DQMFileIterator fiterator_;

    struct OpenFile {
      std::unique_ptr<edm::StreamerInputFile> streamFile_;
      DQMFileIterator::LumiEntry lumi_;

      bool open() { return (streamFile_.get() != nullptr); }

    } file_;

    std::shared_ptr<edm::EventSkipperByID> eventSkipperByID_;
    std::shared_ptr<TriggerSelector> eventSelector_;

    /* this is for monitoring */
    edm::Service<DQMMonitoringService> mon_;
  };

}  // namespace dqmservices

#endif
