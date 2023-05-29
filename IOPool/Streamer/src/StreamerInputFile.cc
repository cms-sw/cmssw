#include "IOPool/Streamer/interface/StreamerInputFile.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Sources/interface/EventSkipperByID.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"

#include "Utilities/StorageFactory/interface/IOFlags.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"

#include <iomanip>
#include <iostream>

namespace edm {

  StreamerInputFile::~StreamerInputFile() { closeStreamerFile(); }

  StreamerInputFile::StreamerInputFile(std::string const& name,
                                       std::string const& LFN,
                                       std::shared_ptr<EventSkipperByID> eventSkipperByID,
                                       unsigned int prefetchMBytes)
      : startMsg_(),
        currentEvMsg_(),
        headerBuf_(1000 * 1000),
        eventBuf_(1000 * 1000 * 7),
        tempBuf_(1024 * 1024 * prefetchMBytes),
        currentFile_(0),
        streamerNames_(),
        multiStreams_(false),
        currentFileName_(),
        currentFileOpen_(false),
        eventSkipperByID_(eventSkipperByID),
        currRun_(0),
        currProto_(0),
        newHeader_(false),
        storage_(),
        endOfFile_(false) {
    openStreamerFile(name, LFN);
    readStartMessage();
  }

  StreamerInputFile::StreamerInputFile(std::string const& name,
                                       std::shared_ptr<EventSkipperByID> eventSkipperByID,
                                       unsigned int prefetchMBytes)
      : StreamerInputFile(name, name, eventSkipperByID, prefetchMBytes) {}

  StreamerInputFile::StreamerInputFile(std::vector<FileCatalogItem> const& names,
                                       std::shared_ptr<EventSkipperByID> eventSkipperByID,
                                       unsigned int prefetchMBytes)
      : startMsg_(),
        currentEvMsg_(),
        headerBuf_(1000 * 1000),
        eventBuf_(1000 * 1000 * 7),
        tempBuf_(1024 * 1024 * prefetchMBytes),
        currentFile_(0),
        streamerNames_(names),
        multiStreams_(true),
        currentFileName_(),
        currentFileOpen_(false),
        eventSkipperByID_(eventSkipperByID),
        currRun_(0),
        currProto_(0),
        newHeader_(false),
        endOfFile_(false) {
    openStreamerFile(names.at(0).fileName(0), names.at(0).logicalFileName());
    ++currentFile_;
    readStartMessage();
    currRun_ = startMsg_->run();
    currProto_ = startMsg_->protocolVersion();
  }

  void StreamerInputFile::openStreamerFile(std::string const& name, std::string const& LFN) {
    closeStreamerFile();

    currentFileName_ = name;

    // Check if the logical file name was found.
    if (currentFileName_.empty()) {
      // LFN not found in catalog.
      throw cms::Exception("LogicalFileNameNotFound", "StreamerInputFile::openStreamerFile()\n")
          << "Logical file name '" << LFN << "' was not found in the file catalog.\n"
          << "If you wanted a local file, you forgot the 'file:' prefix\n"
          << "before the file name in your configuration file.\n";
      return;
    }

    logFileAction("  Initiating request to open file ");

    using namespace edm::storage;
    IOOffset size = -1;
    if (StorageFactory::get()->check(name, &size)) {
      try {
        storage_ = StorageFactory::get()->open(name, IOFlags::OpenRead);
      } catch (cms::Exception& e) {
        Exception ex(errors::FileOpenError, "", e);
        ex.addContext("Calling StreamerInputFile::openStreamerFile()");
        ex.clearMessage();
        ex << "Error Opening Streamer Input File: " << name << "\n";
        throw ex;
      }
    } else {
      throw Exception(errors::FileOpenError, "StreamerInputFile::openStreamerFile")
          << "Error Opening Streamer Input File, file does not exist: " << name << "\n";
    }
    currentFileOpen_ = true;
    logFileAction("  Successfully opened file ");
  }

  void StreamerInputFile::closeStreamerFile() {
    if (currentFileOpen_ && storage_) {
      storage_->close();
      logFileAction("  Closed file ");
    }
    currentFileOpen_ = false;
  }

  std::pair<storage::IOSize, char*> StreamerInputFile::readBytes(char* buf,
                                                                 storage::IOSize nBytes,
                                                                 bool zeroCopy,
                                                                 unsigned int skippedHdr) {
    storage::IOSize n = 0;
    //returned pointer should point to the beginning of the header
    //even if we read event payload that comes afterwards
    char* ptr = buf - skippedHdr;
    try {
      if (!tempBuf_.empty()) {
        if (tempPos_ == tempLen_) {
          n = storage_->read(&tempBuf_[0], tempBuf_.size());
          tempPos_ = 0;
          tempLen_ = n;
          if (n == 0)
            return std::pair<storage::IOSize, char*>(0, ptr);
        }
        if (nBytes <= tempLen_ - tempPos_) {
          //zero-copy can't done when header start address is in the previous buffer
          if (!zeroCopy || skippedHdr > tempPos_) {
            memcpy(buf, &tempBuf_[0] + tempPos_, nBytes);
            tempPos_ += nBytes;
          } else {
            //pass pointer to the prebuffer address (zero copy) at the beginning of the header
            ptr = &tempBuf_[0] + tempPos_ - skippedHdr;
            tempPos_ += nBytes;
          }
          n = nBytes;
        } else {
          //crossing buffer boundary
          auto len = tempLen_ - tempPos_;
          memcpy(buf, &tempBuf_[0] + tempPos_, len);
          tempPos_ += len;
          char* tmpPtr = buf + len;
          n = len + readBytes(tmpPtr, nBytes - len, false).first;
        }
      } else
        n = storage_->read(buf, nBytes);
    } catch (cms::Exception& ce) {
      Exception ex(errors::FileReadError, "", ce);
      ex.addContext("Calling StreamerInputFile::readBytes()");
      throw ex;
    }
    return std::pair<storage::IOSize, char*>(n, ptr);
  }

  storage::IOOffset StreamerInputFile::skipBytes(storage::IOSize nBytes) {
    storage::IOOffset n = 0;
    try {
      // We wish to return the number of bytes skipped, not the final offset.
      n = storage_->position(0, storage::Storage::CURRENT);
      n = storage_->position(nBytes, storage::Storage::CURRENT) - n;
    } catch (cms::Exception& ce) {
      Exception ex(errors::FileReadError, "", ce);
      ex.addContext("Calling StreamerInputFile::skipBytes()");
      throw ex;
    }
    return n;
  }

  void StreamerInputFile::readStartMessage() {
    using namespace edm::storage;
    IOSize nWant = sizeof(InitHeader);
    IOSize nGot = readBytes(&headerBuf_[0], nWant, false).first;
    if (nGot != nWant) {
      throw Exception(errors::FileReadError, "StreamerInputFile::readStartMessage")
          << "Failed reading streamer file, first read in readStartMessage\n";
    }

    uint32 headerSize;
    {
      HeaderView head(&headerBuf_[0]);
      uint32 code = head.code();
      if (code != Header::INIT) /** Not an init message should return ******/
      {
        throw Exception(errors::FileReadError, "StreamerInputFile::readStartMessage")
            << "Expecting an init Message at start of file\n";
        return;
      }
      headerSize = head.size();
    }

    if (headerBuf_.size() < headerSize)
      headerBuf_.resize(headerSize);

    if (headerSize > sizeof(InitHeader)) {
      nWant = headerSize - sizeof(InitHeader);
      auto res = readBytes(&headerBuf_[sizeof(InitHeader)], nWant, true, sizeof(InitHeader));
      if (res.first != nWant) {
        throw Exception(errors::FileReadError, "StreamerInputFile::readStartMessage")
            << "Failed reading streamer file, second read in readStartMessage\n";
      }
      startMsg_ = std::make_shared<InitMsgView>(res.second);  // propagate_const<T> has no reset() function
    } else {
      throw Exception(errors::FileReadError, "StreamerInputFile::readStartMessage")
          << "Failed reading streamer file, init header size from data too small\n";
    }
  }

  StreamerInputFile::Next StreamerInputFile::next() {
    if (this->readEventMessage()) {
      return Next::kEvent;
    }
    if (multiStreams_) {
      //Try opening next file
      if (currentFile_ <= streamerNames_.size() - 1) {
        newHeader_ = true;
        return Next::kFile;
      }
    }
    return Next::kStop;
  }

  bool StreamerInputFile::openNextFile() {
    if (currentFile_ <= streamerNames_.size() - 1) {
      FDEBUG(10) << "Opening file " << streamerNames_.at(currentFile_).fileNames()[0].c_str() << std::endl;

      openStreamerFile(streamerNames_.at(currentFile_).fileNames()[0],
                       streamerNames_.at(currentFile_).logicalFileName());

      // If start message was already there, then compare the
      // previous and new headers
      if (startMsg_) {
        FDEBUG(10) << "Comparing Header" << std::endl;
        compareHeader();
      }
      ++currentFile_;
      endOfFile_ = false;
      return true;
    }
    return false;
  }

  bool StreamerInputFile::compareHeader() {
    //Get the new header
    readStartMessage();

    //Values from new Header should match up
    if (currRun_ != startMsg_->run() || currProto_ != startMsg_->protocolVersion()) {
      throw Exception(errors::MismatchedInputFiles, "StreamerInputFile::compareHeader")
          << "File " << streamerNames_.at(currentFile_).fileNames()[0]
          << "\nhas different run number or protocol version than previous\n";
      return false;
    }
    return true;
  }

  int StreamerInputFile::readEventMessage() {
    if (endOfFile_)
      return 0;

    using namespace edm::storage;
    bool eventRead = false;
    unsigned hdrSkipped = 0;
    while (!eventRead) {
      IOSize nWant = sizeof(EventHeader);
      IOSize nGot = readBytes(&eventBuf_[hdrSkipped], nWant - hdrSkipped, false).first + hdrSkipped;
      while (nGot == nWant) {
        //allow padding before next event or end of file.
        //event header starts with code 0 - 17, so 0xff (Header:PADDING) uniquely represents padding
        bool headerFetched = false;
        for (size_t i = 0; i < nGot; i++) {
          if ((unsigned char)eventBuf_[i] != Header::PADDING) {
            //no padding 0xff
            if (i != 0) {
              memmove(&eventBuf_[0], &eventBuf_[i], nGot - i);
              //read remainder of the header
              nGot = nGot - i + readBytes(&eventBuf_[nGot - i], i, false).first;
            }
            headerFetched = true;
            break;
          }
        }
        if (headerFetched)
          break;
        //read another block
        nGot = readBytes(&eventBuf_[0], nWant, false).first;
      }
      if (nGot == 0) {
        // no more data available
        endOfFile_ = true;
        return 0;
      }
      if (nGot != nWant) {
        for (size_t i = 0; i < nGot; i++) {
          if ((unsigned char)eventBuf_[i] != Header::PADDING)
            throw edm::Exception(errors::FileReadError, "StreamerInputFile::readEventMessage")
                << "Failed reading streamer file, first read in readEventMessage\n"
                << "Requested " << nWant << " bytes, read function returned " << nGot
                << " bytes, non-padding at offset " << i;
        }
        //padded 0xff only
        endOfFile_ = true;
        return 0;
      }
      uint32 eventSize;
      {
        HeaderView head(&eventBuf_[0]);
        uint32 code = head.code();

        // If it is not an event then something is wrong.
        eventSize = head.size();
        if (code != Header::EVENT) {
          if (code == Header::INIT) {
            edm::LogWarning("StreamerInputFile") << "Found another INIT header in the file. It will be skipped";
            if (eventSize < sizeof(EventHeader)) {
              //very unlikely case that EventHeader is larger than total INIT size inserted in the middle of the file
              hdrSkipped = nGot - eventSize;
              memmove(&eventBuf_[0], &eventBuf_[eventSize], hdrSkipped);
              continue;
            }
            if (headerBuf_.size() < eventSize)
              headerBuf_.resize(eventSize);
            memcpy(&headerBuf_[0], &eventBuf_[0], nGot);
            readBytes(&headerBuf_[nGot], eventSize, true, nGot);
            //do not parse this header and proceed to the next event
            continue;
          }
          throw Exception(errors::FileReadError, "StreamerInputFile::readEventMessage")
              << "Failed reading streamer file, unknown code in event header\n"
              << "code = " << code << "\n";
        }
      }
      if (eventSize <= sizeof(EventHeader)) {
        throw edm::Exception(errors::FileReadError, "StreamerInputFile::readEventMessage")
            << "Failed reading streamer file, event header size from data too small\n";
      }
      eventRead = true;
      if (eventSkipperByID_) {
        EventHeader* evh = (EventHeader*)(&eventBuf_[0]);
        if (eventSkipperByID_->skipIt(convert32(evh->run_), convert32(evh->lumi_), convert64(evh->event_))) {
          eventRead = false;
        }
      }
      nWant = eventSize - sizeof(EventHeader);
      if (eventRead) {
        if (eventBuf_.size() < eventSize)
          eventBuf_.resize(eventSize);

        auto res = readBytes(&eventBuf_[sizeof(EventHeader)], nWant, true, sizeof(EventHeader));
        if (res.first != nWant) {
          throw Exception(errors::FileReadError, "StreamerInputFile::readEventMessage")
              << "Failed reading streamer file, second read in readEventMessage\n"
              << "Requested " << nWant << " bytes, read function returned " << res.first << " bytes\n";
        }
        currentEvMsg_ =
            std::make_shared<EventMsgView>((void*)res.second);  // propagate_const<T> has no reset() function
      } else {
        nGot = skipBytes(nWant);
        if (nGot != nWant) {
          throw Exception(errors::FileReadError, "StreamerInputFile::readEventMessage")
              << "Failed reading streamer file, skip event in readEventMessage\n"
              << "Requested " << nWant << " bytes skipped, seek function returned " << nGot << " bytes\n";
        }
      }
    }
    return 1;
  }

  void StreamerInputFile::logFileAction(char const* msg) {
    LogAbsolute("fileAction") << std::setprecision(0) << TimeOfDay() << msg << currentFileName_;
    FlushMessageLog();
  }
}  // namespace edm
