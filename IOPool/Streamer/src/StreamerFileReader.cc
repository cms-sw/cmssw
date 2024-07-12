#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/StreamerInputFile.h"
#include "IOPool/Streamer/src/StreamerFileReader.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Sources/interface/EventSkipperByID.h"

#include <cassert>
namespace edm::streamer {

  StreamerFileReader::StreamerFileReader(ParameterSet const& pset, InputSourceDescription const& desc)
      : StreamerInputSource(pset, desc),
        streamReader_(),
        eventSkipperByID_(EventSkipperByID::create(pset).release()),
        initialNumberOfEventsToSkip_(pset.getUntrackedParameter<unsigned int>("skipEvents")),
        prefetchMBytes_(pset.getUntrackedParameter<unsigned int>("prefetchMBytes")) {
    InputFileCatalog catalog(pset.getUntrackedParameter<std::vector<std::string> >("fileNames"),
                             pset.getUntrackedParameter<std::string>("overrideCatalog"));
    streamerNames_ = catalog.fileCatalogItems();
    reset_();
  }

  StreamerFileReader::~StreamerFileReader() {}

  void StreamerFileReader::reset_() {
    if (streamerNames_.size() > 1) {
      streamReader_ = std::make_unique<StreamerInputFile>(streamerNames_, eventSkipperByID(), prefetchMBytes_);
    } else if (streamerNames_.size() == 1) {
      streamReader_ = std::make_unique<StreamerInputFile>(streamerNames_.at(0).fileNames()[0],
                                                          streamerNames_.at(0).logicalFileName(),
                                                          eventSkipperByID(),
                                                          prefetchMBytes_);
    } else {
      throw Exception(errors::FileReadError, "StreamerFileReader::StreamerFileReader")
          << "No fileNames were specified\n";
    }
    isFirstFile_ = true;
    updateMetaData(false);
    if (initialNumberOfEventsToSkip_) {
      skip(initialNumberOfEventsToSkip_);
    }
  }

  void StreamerFileReader::updateMetaData(bool subsequent) {
    InitMsgView const* header = getHeader();
    deserializeAndMergeWithRegistry(*header, subsequent);
    //NOTE: should read first Event to get the meta data and then set 'artificial file'
    auto eview = getNextEvent();

    //if no events then file must be empty
    if (eview == nullptr)
      return;

    assert(eview->isEventMetaData());
    deserializeEventMetaData(*eview);
    updateEventMetaData();
  }

  StreamerFileReader::Next StreamerFileReader::checkNext() {
    EventMsgView const* eview = getNextEvent();

    if (eview == nullptr) {
      if (newHeader()) {
        return Next::kFile;
      }
      return Next::kStop;
    }
    if (eview->isEventMetaData()) {
      if (presentEventMetaDataChecksum() != eventMetaDataChecksum(*eview)) {
        //we lie and say there is a new file since we need to synchronize to update the meta data
        didArtificialFile_ = true;
        deserializeEventMetaData(*eview);
        return Next::kFile;
      } else {
        //skip this meta data
        eview = getNextEvent();
        if (eview == nullptr) {
          if (newHeader()) {
            return Next::kFile;
          }
          return Next::kStop;
        }
      }
    }
    deserializeEvent(*eview);
    return Next::kEvent;
  }

  void StreamerFileReader::skip(int toSkip) {
    for (int i = 0; i != toSkip; ++i) {
      EventMsgView const* evMsg = getNextEvent();
      if (evMsg == nullptr) {
        return;
      }
      // If the event would have been skipped anyway, don't count it as a skipped event.
      if (eventSkipperByID_ && eventSkipperByID_->skipIt(evMsg->run(), evMsg->lumi(), evMsg->event())) {
        --i;
      }
    }
  }

  void StreamerFileReader::genuineCloseFile() {
    if (didArtificialFile_) {
      return;
    }
    if (streamReader_.get() != nullptr)
      streamReader_->closeStreamerFile();
  }

  void StreamerFileReader::genuineReadFile() {
    if (isFirstFile_) {
      //The file was already opened in the constructor
      isFirstFile_ = false;
      return;
    }
    if (didArtificialFile_) {
      //update the event meta data
      didArtificialFile_ = false;
      updateEventMetaData();
      return;
    }
    streamReader_->openNextFile();
    // FDEBUG(6) << "A new file has been opened and we must compare Headers here !!" << std::endl;
    // A new file has been opened and we must compare Heraders here !!
    //Get header/init from reader
    updateMetaData(true);
  }

  bool StreamerFileReader::newHeader() { return streamReader_->newHeader(); }

  InitMsgView const* StreamerFileReader::getHeader() {
    InitMsgView const* header = streamReader_->startMessage();

    if (header->code() != Header::INIT) {  //INIT Msg
      throw Exception(errors::FileReadError, "StreamerFileReader::readHeader")
          << "received wrong message type: expected INIT, got " << header->code() << "\n";
    }
    return header;
  }

  EventMsgView const* StreamerFileReader::getNextEvent() {
    if (StreamerInputFile::Next::kEvent != streamReader_->next()) {
      return nullptr;
    }
    return streamReader_->currentRecord();
  }

  void StreamerFileReader::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setComment("Reads events from streamer files.");
    desc.addUntracked<std::vector<std::string> >("fileNames")->setComment("Names of files to be processed.");
    desc.addUntracked<unsigned int>("skipEvents", 0U)
        ->setComment("Skip the first 'skipEvents' events that otherwise would have been processed.");
    desc.addUntracked<std::string>("overrideCatalog", std::string());
    //This next parameter is read in the base class, but its default value depends on the derived class, so it is set here.
    desc.addUntracked<bool>("inputFileTransitionsEachEvent", false);
    desc.addUntracked<unsigned int>("prefetchMBytes", 0);
    StreamerInputSource::fillDescription(desc);
    EventSkipperByID::fillDescription(desc);
    descriptions.add("source", desc);
  }
}  // namespace edm::streamer
