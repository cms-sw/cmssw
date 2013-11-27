#include "IOPool/Streamer/interface/StreamerInputFile.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Sources/interface/EventSkipperByID.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"

#include "Utilities/StorageFactory/interface/IOFlags.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"

#include <iomanip>
#include <iostream>

namespace edm {

  StreamerInputFile::~StreamerInputFile() {
    closeStreamerFile();
  }

  StreamerInputFile::StreamerInputFile(std::string const& name,
                                       boost::shared_ptr<EventSkipperByID> eventSkipperByID) :
    startMsg_(),
    currentEvMsg_(),
    headerBuf_(1000*1000),
    eventBuf_(1000*1000*7),
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
    openStreamerFile(name);
    readStartMessage();
  }

  StreamerInputFile::StreamerInputFile(std::vector<std::string> const& names,
                                       boost::shared_ptr<EventSkipperByID> eventSkipperByID) :
    startMsg_(),
    currentEvMsg_(),
    headerBuf_(1000*1000),
    eventBuf_(1000*1000*7),
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
    openStreamerFile(names.at(0));
    ++currentFile_;
    readStartMessage();
    currRun_ = startMsg_->run();
    currProto_ = startMsg_->protocolVersion();
  }

  void
  StreamerInputFile::openStreamerFile(std::string const& name) {

    closeStreamerFile();

    currentFileName_ = name;
    logFileAction("  Initiating request to open file ");

    IOOffset size = -1;
    if(StorageFactory::get()->check(name.c_str(), &size)) {
      try {
        storage_.reset(StorageFactory::get()->open(name.c_str(),
                                                   IOFlags::OpenRead));
      }
      catch(cms::Exception& e) {
        Exception ex(errors::FileOpenError, "", e);
        ex.addContext("Calling StreamerInputFile::openStreamerFile()");
        ex.clearMessage();
        ex <<  "Error Opening Streamer Input File: " << name << "\n";
        throw ex;
      }
    } else {
      throw Exception(errors::FileOpenError, "StreamerInputFile::openStreamerFile")
        << "Error Opening Streamer Input File, file does not exist: "
        << name << "\n";
    }
    currentFileOpen_ = true;
    logFileAction("  Successfully opened file ");
  }

  void
  StreamerInputFile::closeStreamerFile() {
    if(currentFileOpen_ && storage_) {
      storage_->close();
      logFileAction("  Closed file ");
    }
    currentFileOpen_ = false;
  }

  IOSize StreamerInputFile::readBytes(char *buf, IOSize nBytes) {
    IOSize n = 0;
    try {
      n = storage_->read(buf, nBytes);
    }
    catch(cms::Exception& ce) {
      Exception ex(errors::FileReadError, "", ce);
      ex.addContext("Calling StreamerInputFile::readBytes()");
      throw ex;
    }
    return n;
  }

  IOOffset StreamerInputFile::skipBytes(IOSize nBytes) {
    IOOffset n = 0;
    try {
      // We wish to return the number of bytes skipped, not the final offset.
      n = storage_->position(0, Storage::CURRENT);
      n = storage_->position(nBytes, Storage::CURRENT) - n;
    }
    catch(cms::Exception& ce) {
      Exception ex(errors::FileReadError, "", ce);
      ex.addContext("Calling StreamerInputFile::skipBytes()");
      throw ex;
    }
    return n;
  }

  void StreamerInputFile::readStartMessage() {
    IOSize nWant = sizeof(HeaderView);
    IOSize nGot = readBytes(&headerBuf_[0], nWant);
    if(nGot != nWant) {
      throw Exception(errors::FileReadError, "StreamerInputFile::readStartMessage")
        << "Failed reading streamer file, first read in readStartMessage\n";
    }

    HeaderView head(&headerBuf_[0]);
    uint32 code = head.code();
    if(code != Header::INIT) /** Not an init message should return ******/
    {
      throw Exception(errors::FileReadError, "StreamerInputFile::readStartMessage")
        << "Expecting an init Message at start of file\n";
      return;
    }

    uint32 headerSize = head.size();
    if(headerBuf_.size() < headerSize) headerBuf_.resize(headerSize);

    if(headerSize > sizeof(HeaderView)) {
      nWant = headerSize - sizeof(HeaderView);
      nGot = readBytes(&headerBuf_[sizeof(HeaderView)], nWant);
      if(nGot != nWant) {
        throw Exception(errors::FileReadError, "StreamerInputFile::readStartMessage")
          << "Failed reading streamer file, second read in readStartMessage\n";
      }
    } else {
      throw Exception(errors::FileReadError, "StreamerInputFile::readStartMessage")
        << "Failed reading streamer file, init header size from data too small\n";
    }

    startMsg_.reset(new InitMsgView(&headerBuf_[0]));
  }

  bool StreamerInputFile::next() {
    if(this->readEventMessage()) {
         return true;
    }
    if(multiStreams_) {
       //Try opening next file
       if(openNextFile()) {
          endOfFile_ = false;
          if(this->readEventMessage()) {
             return true;
          }
       }
    }
    return false;
  }

  bool StreamerInputFile::openNextFile() {

     if(currentFile_ <= streamerNames_.size() - 1) {
       FDEBUG(10) << "Opening file "
                  << streamerNames_.at(currentFile_).c_str() << std::endl;

       openStreamerFile(streamerNames_.at(currentFile_));

       // If start message was already there, then compare the
       // previous and new headers
       if(startMsg_) {
          FDEBUG(10) << "Comparing Header" << std::endl;
          if(!compareHeader()) {
              return false;
          }
       }
       ++currentFile_;
       return true;
     }
     return false;
  }

  bool StreamerInputFile::compareHeader() {

    //Get the new header
    readStartMessage();

    //Values from new Header should match up
    if(currRun_ != startMsg_->run() ||
        currProto_ != startMsg_->protocolVersion()) {
      throw Exception(errors::MismatchedInputFiles,"StreamerInputFile::compareHeader")
        << "File " << streamerNames_.at(currentFile_)
        << "\nhas different run number or protocol version than previous\n";
      return false;
    }
    newHeader_ = true;
    return true;
  }


  int StreamerInputFile::readEventMessage() {
    if(endOfFile_) return 0;

    bool eventRead = false;
    while(!eventRead) {

      IOSize nWant = sizeof(EventHeader);
      IOSize nGot = readBytes(&eventBuf_[0], nWant);
      if(nGot != nWant) {
        throw edm::Exception(errors::FileReadError, "StreamerInputFile::readEventMessage")
          << "Failed reading streamer file, first read in readEventMessage\n"
          << "Requested " << nWant << " bytes, read function returned " << nGot << " bytes\n";
      }
      HeaderView head(&eventBuf_[0]);
      uint32 code = head.code();

      // When we get the EOF record we know we have read all events
      // normally and are at the end, return 0 to indicate this
      if(code == Header::EOFRECORD) {
        endOfFile_ = true;
        return 0;
      }
      // If it is not an event nor EOFRECORD then something is wrong.
      if(code != Header::EVENT) {
        throw Exception(errors::FileReadError, "StreamerInputFile::readEventMessage")
          << "Failed reading streamer file, unknown code in event header\n"
          << "code = " << code << "\n";
      }
      uint32 eventSize = head.size();
      if(eventSize <= sizeof(EventHeader)) {
        throw edm::Exception(errors::FileReadError, "StreamerInputFile::readEventMessage")
          << "Failed reading streamer file, event header size from data too small\n";
      }
      eventRead = true;
      if(eventSkipperByID_) {
        EventHeader *evh = (EventHeader *)(&eventBuf_[0]);
        if(eventSkipperByID_->skipIt(convert32(evh->run_), convert32(evh->lumi_), convert32(evh->event_))) {
          eventRead = false;
        }
      }
      nWant = eventSize - sizeof(EventHeader);
      if(eventRead) {
        if(eventBuf_.size() < eventSize) eventBuf_.resize(eventSize);
        nGot = readBytes(&eventBuf_[sizeof(EventHeader)], nWant);
        if(nGot != nWant) {
          throw Exception(errors::FileReadError, "StreamerInputFile::readEventMessage")
            << "Failed reading streamer file, second read in readEventMessage\n"
            << "Requested " << nWant << " bytes, read function returned " << nGot << " bytes\n";
        }
      } else {
        nGot = skipBytes(nWant);
        if(nGot != nWant) {
          throw Exception(errors::FileReadError, "StreamerInputFile::readEventMessage")
            << "Failed reading streamer file, skip event in readEventMessage\n"
            << "Requested " << nWant << " bytes skipped, seek function returned " << nGot << " bytes\n";
        }
      }
    }
    currentEvMsg_.reset(new EventMsgView((void*)&eventBuf_[0]));
    return 1;
  }

  void StreamerInputFile::logFileAction(char const* msg) {
    LogAbsolute("fileAction") << std::setprecision(0) << TimeOfDay() << msg << currentFileName_;
    FlushMessageLog();
  }
}
