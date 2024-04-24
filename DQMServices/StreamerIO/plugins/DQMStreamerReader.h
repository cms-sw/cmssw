#ifndef DQMServices_StreamerIO_DQMStreamerReader_h
#define DQMServices_StreamerIO_DQMStreamerReader_h

#include "IOPool/Streamer/interface/StreamerInputSource.h"
#include "IOPool/Streamer/interface/StreamerInputFile.h"
#include "IOPool/Streamer/interface/MsgTools.h"

#include "DQMFileIterator.h"
#include "TriggerSelector.h"

#include <memory>
#include <string>
#include <vector>

namespace dqmservices {

  class DQMStreamerReader : public edm::streamer::StreamerInputSource {
  public:
    DQMStreamerReader(edm::ParameterSet const& pset, edm::InputSourceDescription const& desc);
    ~DQMStreamerReader() override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    bool newHeader();

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

    edm::streamer::InitMsgView const* getHeaderMsg();
    edm::streamer::EventMsgView const* getEventMsg();

    void setupMetaData(edm::streamer::InitMsgView const& msg, bool subsequent);
    edm::streamer::EventMsgView const* prepareNextEvent();

    bool isFirstFile_ = true;
    bool prepareNextFile();
    bool acceptEvent(const edm::streamer::EventMsgView*);

    DQMFileIterator fiterator_;
    unsigned int processedEventPerLs_ = 0;

    unsigned int const minEventsPerLs_;
    bool const flagSkipFirstLumis_;
    bool const flagEndOfRunKills_;
    bool const flagDeleteDatFiles_;
    std::vector<std::string> const hltSel_;

    bool acceptAllEvt_ = false;
    bool setAcceptAllEvt();

    bool matchTriggerSel_ = false;
    bool setMatchTriggerSel(std::vector<std::string> const& tnames);

    //If the event meta data changes while reading a file, we need to
    // cause a file transition to happen to allow synchronous update
    bool artificialFileBoundary_ = false;
    struct OpenFile {
      std::unique_ptr<edm::streamer::StreamerInputFile> streamFile_;
      DQMFileIterator::LumiEntry lumi_;

      bool open() { return (streamFile_.get() != nullptr); }

    } file_;

    std::shared_ptr<edm::EventSkipperByID> eventSkipperByID_;
    std::shared_ptr<TriggerSelector> triggerSelector_;
  };

}  // namespace dqmservices

#endif  // DQMServices_StreamerIO_DQMStreamerReader_h
