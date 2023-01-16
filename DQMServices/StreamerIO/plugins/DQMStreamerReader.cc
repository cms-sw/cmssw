#include "DQMStreamerReader.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Sources/interface/EventSkipperByID.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/RegexMatch.h"
#include "FWCore/Utilities/interface/UnixSignalHandlers.h"
#include "IOPool/Streamer/interface/DumpTools.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <queue>
#include <algorithm>
#include <cctype>

namespace dqmservices {

  DQMStreamerReader::DQMStreamerReader(edm::ParameterSet const& pset, edm::InputSourceDescription const& desc)
      : StreamerInputSource(pset, desc),
        fiterator_(pset),
        minEventsPerLs_(pset.getUntrackedParameter<int>("minEventsPerLumi")),
        flagSkipFirstLumis_(pset.getUntrackedParameter<bool>("skipFirstLumis")),
        flagEndOfRunKills_(pset.getUntrackedParameter<bool>("endOfRunKills")),
        flagDeleteDatFiles_(pset.getUntrackedParameter<bool>("deleteDatFiles")),
        hltSel_(pset.getUntrackedParameter<std::vector<std::string>>("SelectEvents")) {
    setAcceptAllEvt();
    reset_();
  }

  DQMStreamerReader::~DQMStreamerReader() {
    // Sometimes(?) the destructor called after service registry was already destructed
    // and closeFile_ throws away no ServiceRegistry found exception...
    //
    // Normally, this file should be closed before this destructor is called.
    //closeFileImp_("destructor");
  }

  void DQMStreamerReader::reset_() {
    // We have to load at least a single header,
    // so the ProductRegistry gets initialized.
    //
    // This must happen here (inside the constructor),
    // as ProductRegistry gets frozen after we initialize:
    // https://cmssdt.cern.ch/SDT/lxr/source/FWCore/Framework/src/Schedule.cc#441

    fiterator_.logFileAction("Waiting for the first lumi in order to initialize.");

    fiterator_.update_state();

    // Fast-forward to the last open file.
    if (flagSkipFirstLumis_) {
      unsigned int l = fiterator_.lastLumiFound();
      if (l > 1) {
        fiterator_.advanceToLumi(l, "skipped: fast-forward to the latest lumi");
      }
    }

    for (;;) {
      bool next = prepareNextFile();

      // check for end of run
      if (!next) {
        fiterator_.logFileAction("End of run reached before DQMStreamerReader was initialised.");
        return;
      }

      // check if we have a file openned
      if (file_.open()) {
        // we are now initialised
        break;
      }

      // wait
      fiterator_.delay();
    }

    fiterator_.logFileAction("DQMStreamerReader initialised.");
  }

  void DQMStreamerReader::openFileImp_(const DQMFileIterator::LumiEntry& entry) {
    processedEventPerLs_ = 0;

    std::string path = entry.get_data_path();

    file_.lumi_ = entry;
    file_.streamFile_ = std::make_unique<edm::StreamerInputFile>(path);

    InitMsgView const* header = getHeaderMsg();
    if (isFirstFile_) {
      deserializeAndMergeWithRegistry(*header, false);
    }

    // dump the list of HLT trigger name from the header
    //  dumpInitHeader(header);

    // if specific trigger selection is requested, check if the requested triggers match with trigger paths in the header file
    if (!acceptAllEvt_) {
      std::vector<std::string> tnames;
      header->hltTriggerNames(tnames);

      triggerSelector_.reset(new TriggerSelector(hltSel_, tnames));

      // check if any trigger path name requested matches with trigger name in the header file
      setMatchTriggerSel(tnames);
    }

    // our initialization
    processedEventPerLs_ = 0;

    if (flagDeleteDatFiles_) {
      // unlink the file
      unlink(path.c_str());
    }
  }

  void DQMStreamerReader::genuineCloseFile() {}

  void DQMStreamerReader::closeFileImp_(const std::string& reason) {
    if (file_.open()) {
      file_.streamFile_->closeStreamerFile();
      file_.streamFile_ = nullptr;

      fiterator_.logLumiState(file_.lumi_, "close: " + reason);
    }
  }

  void DQMStreamerReader::genuineReadFile() {
    if (isFirstFile_) {
      //The file was already opened in the constructor
      isFirstFile_ = false;
      return;
    }

    //Get header/init from reader
    InitMsgView const* header = getHeaderMsg();
    deserializeAndMergeWithRegistry(*header, true);
  }

  bool DQMStreamerReader::openNextFileImp_() {
    closeFileImp_("skipping to another file");

    DQMFileIterator::LumiEntry currentLumi = fiterator_.open();
    std::string p = currentLumi.get_data_path();

    if (std::filesystem::exists(p)) {
      try {
        openFileImp_(currentLumi);
        return true;
      } catch (const cms::Exception& e) {
        fiterator_.logFileAction(std::string("Can't deserialize registry data (in open file): ") + e.what(), p);
        fiterator_.logLumiState(currentLumi, "error: data file corrupted");

        closeFileImp_("data file corrupted");
        return false;
      }
    } else {
      /* dat file missing */
      fiterator_.logFileAction("Data file (specified in json) is missing:", p);
      fiterator_.logLumiState(currentLumi, "error: data file missing");

      return false;
    }
  }

  InitMsgView const* DQMStreamerReader::getHeaderMsg() {
    InitMsgView const* header = file_.streamFile_->startMessage();

    if (header->code() != Header::INIT) {  // INIT Msg
      throw edm::Exception(edm::errors::FileReadError, "DQMStreamerReader::readHeader")
          << "received wrong message type: expected INIT, got " << header->code() << "\n";
    }

    return header;
  }

  EventMsgView const* DQMStreamerReader::getEventMsg() {
    auto next = file_.streamFile_->next();
    if (edm::StreamerInputFile::Next::kFile == next) {
      return nullptr;
    }

    if (edm::StreamerInputFile::Next::kStop == next) {
      return nullptr;
    }

    EventMsgView const* msg = file_.streamFile_->currentRecord();

    //  if (msg != nullptr) dumpEventView(msg);
    return msg;
  }

  /**
 * Prepare (open) the next file for reading.
 * It is used by prepareNextEvent and in the constructor.
 *
 * Does not block/wait.
 *
 * Return false if this is end of run and/or no more file are available.
 * However, return of "true" does not imply the file has been openned,
 * but we need to wait until some future file becomes available.
 */
  bool DQMStreamerReader::prepareNextFile() {
    typedef DQMFileIterator::State State;

    for (;;) {
      fiterator_.update_state();

      if (edm::shutdown_flag.load()) {
        fiterator_.logFileAction("Shutdown flag was set, shutting down.");

        closeFileImp_("shutdown flag is set");
        return false;
      }

      // check for end of run file and force quit
      if (flagEndOfRunKills_ && (fiterator_.state() != State::OPEN)) {
        closeFileImp_("forced end-of-run");
        return false;
      }

      // check for end of run and quit if everything has been processed.
      // this clean exit
      if ((!file_.open()) && (!fiterator_.lumiReady()) && (fiterator_.state() == State::EOR)) {
        return false;
      }

      // if this is end of run and no more files to process
      // close it
      if ((processedEventPerLs_ >= minEventsPerLs_) && (!fiterator_.lumiReady()) &&
          (fiterator_.state() == State::EOR)) {
        closeFileImp_("graceful end-of-run");
        return false;
      }

      // skip to the next file if we have no files openned yet
      if (!file_.open()) {
        if (fiterator_.lumiReady()) {
          openNextFileImp_();
          // we might need to open once more (if .dat is missing)
          continue;
        }
      }

      // or if there is a next file and enough eventshas been processed.
      if (fiterator_.lumiReady() && (processedEventPerLs_ >= minEventsPerLs_)) {
        openNextFileImp_();
        // we might need to open once more (if .dat is missing)
        continue;
      }

      return true;
    }
  }

  /**
 * Waits and reads the event header.
 * If end-of-run nullptr is returned.
 */
  EventMsgView const* DQMStreamerReader::prepareNextEvent() {
    EventMsgView const* eview = nullptr;
    typedef DQMFileIterator::State State;

    // wait for the next event
    for (;;) {
      // edm::LogAbsolute("DQMStreamerReader")
      //     << "State loop.";
      bool next = prepareNextFile();
      if (!next)
        return nullptr;

      // sleep
      if (!file_.open()) {
        // the reader does not exist
        fiterator_.delay();
      } else {
        // our reader exists, try to read out an event
        eview = getEventMsg();

        if (eview == nullptr) {
          // read unsuccessful
          // this means end of file, so close the file
          closeFileImp_("eof");
        } else {
          if (!acceptEvent(eview)) {
            continue;
          } else {
            return eview;
          }
        }
      }
    }
    return eview;
  }

  /**
 * This is the actual code for checking the new event and/or deserializing it.
 */
  edm::RawInputSource::Next DQMStreamerReader::checkNext() {
    try {
      EventMsgView const* eview = prepareNextEvent();
      if (eview == nullptr) {
        if (file_.streamFile_ and file_.streamFile_->newHeader()) {
          return Next::kFile;
        }
        return Next::kStop;
      }

      deserializeEvent(*eview);
    } catch (const cms::Exception& e) {
      // try to recover from corrupted files/events
      fiterator_.logFileAction(std::string("Can't deserialize event or registry data: ") + e.what());
      closeFileImp_("data file corrupted");

      // this is not optimal, but hopefully we won't catch this many times in a row
      return checkNext();
    }

    processedEventPerLs_ += 1;

    return Next::kEvent;
  }

  /**
 * If hlt trigger selection is '*', return a boolean variable to accept all
 * events
 */
  bool DQMStreamerReader::setAcceptAllEvt() {
    acceptAllEvt_ = false;
    for (auto hltPath : hltSel_) {
      hltPath.erase(std::remove_if(hltPath.begin(), hltPath.end(), [](unsigned char c) { return std::isspace(c); }),
                    hltPath.end());
      if (hltPath == "*") {
        acceptAllEvt_ = true;
        break;
      }
    }
    return acceptAllEvt_;
  }

  /**
 * Check if hlt selection matches any trigger name taken from the header file
 */
  bool DQMStreamerReader::setMatchTriggerSel(std::vector<std::string> const& tnames) {
    matchTriggerSel_ = false;
    for (auto hltPath : hltSel_) {
      hltPath.erase(std::remove_if(hltPath.begin(), hltPath.end(), [](unsigned char c) { return std::isspace(c); }),
                    hltPath.end());
      auto const matches = edm::regexMatch(tnames, hltPath);
      if (not matches.empty()) {
        matchTriggerSel_ = true;
        break;
      }
    }

    if (not matchTriggerSel_) {
      edm::LogWarning("DQMStreamerReader") << "Trigger selection does not match any trigger path!!!";
    }

    return matchTriggerSel_;
  }

  /**
 * Check the trigger path to accept event
 */
  bool DQMStreamerReader::acceptEvent(const EventMsgView* evtmsg) {
    if (acceptAllEvt_)
      return true;
    if (!matchTriggerSel_)
      return false;

    std::vector<unsigned char> hltTriggerBits_;
    int hltTriggerCount_ = evtmsg->hltCount();
    if (hltTriggerCount_ > 0) {
      hltTriggerBits_.resize(1 + (hltTriggerCount_ - 1) / 4);
    }
    evtmsg->hltTriggerBits(&hltTriggerBits_[0]);

    return (triggerSelector_->wantAll() || triggerSelector_->acceptEvent(&hltTriggerBits_[0], evtmsg->hltCount()));
  }

  void DQMStreamerReader::skip(int toSkip) {
    try {
      for (int i = 0; i != toSkip; ++i) {
        EventMsgView const* evMsg = prepareNextEvent();

        if (evMsg == nullptr) {
          return;
        }
      }
    } catch (const cms::Exception& e) {
      // try to recover from corrupted files/events
      fiterator_.logFileAction(std::string("Can't deserialize event data: ") + e.what());
      closeFileImp_("data file corrupted");
    }
  }

  void DQMStreamerReader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.setComment("Reads events from streamer files.");

    desc.addUntracked<std::vector<std::string>>("SelectEvents")->setComment("HLT path to select events");

    desc.addUntracked<int>("minEventsPerLumi", 1)
        ->setComment(
            "Minimum number of events to process per lumisection, "
            "before switching to a new input file. If the next file "
            "does not yet exist, "
            "the number of processed events will be bigger.");

    desc.addUntracked<bool>("skipFirstLumis", false)
        ->setComment(
            "Skip (and ignore the minEventsPerLumi parameter) for the files "
            "which have been available at the begining of the processing. "
            "If set to true, the reader will open last available file for "
            "processing.");

    desc.addUntracked<bool>("deleteDatFiles", false)
        ->setComment(
            "Delete data files after they have been closed, in order to "
            "save disk space.");

    desc.addUntracked<bool>("endOfRunKills", false)
        ->setComment(
            "Kill the processing as soon as the end-of-run file appears, even if "
            "there are/will be unprocessed lumisections.");

    // desc.addUntracked<unsigned int>("skipEvents", 0U)
    //    ->setComment("Skip the first 'skipEvents' events that otherwise would "
    //                 "have been processed.");

    // This next parameter is read in the base class, but its default value
    // depends on the derived class, so it is set here.
    desc.addUntracked<bool>("inputFileTransitionsEachEvent", false);

    DQMFileIterator::fillDescription(desc);
    edm::StreamerInputSource::fillDescription(desc);
    edm::EventSkipperByID::fillDescription(desc);

    descriptions.add("source", desc);
  }

}  // namespace dqmservices

#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef dqmservices::DQMStreamerReader DQMStreamerReader;
DEFINE_FWK_INPUT_SOURCE(DQMStreamerReader);
