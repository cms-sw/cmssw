#include "IOPool/Streamer/interface/StreamerInputFile.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/StreamerInputIndexFile.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "SealBase/IOError.h"

#include <iostream>

using namespace edm;

StreamerInputFile::~StreamerInputFile()
{
  if (storage_) storage_->close();

  delete startMsg_;
  delete  currentEvMsg_;
}

StreamerInputFile::StreamerInputFile(const std::string& name):
  useIndex_(false),
  startMsg_(0),
  currentEvMsg_(0),
  headerBuf_(1000*1000),
  eventBuf_(1000*1000*7),
  multiStreams_(false),
  newHeader_(false),
  endOfFile_(false)
{
  openStreamerFile(name);
  readStartMessage();
}

StreamerInputFile::StreamerInputFile(const std::string& name, 
                                     const std::string& order):
  useIndex_(true),
  index_(new StreamerInputIndexFile(order)),
  //indexIter_b(index_->begin()),
  indexIter_b(index_->sort()),
  indexIter_e(index_->end()),
  startMsg_(0),
  currentEvMsg_(0),
  headerBuf_(1000*1000),
  eventBuf_(1000*1000*7),
  multiStreams_(false),
  newHeader_(false),
  endOfFile_(false)
{
  openStreamerFile(name);
  readStartMessage();
}

StreamerInputFile::StreamerInputFile(const std::string& name,
                                     const StreamerInputIndexFile& order):
  useIndex_(true),
  index_((StreamerInputIndexFile*)&order),
  //indexIter_b(index_->begin()),
  indexIter_b(index_->sort()),
  indexIter_e(index_->end()),
  startMsg_(0),
  currentEvMsg_(0),
  headerBuf_(1000*1000),
  eventBuf_(1000*1000*7),
  multiStreams_(false),
  newHeader_(false),
  endOfFile_(false)
{
  openStreamerFile(name);
  readStartMessage();
}


StreamerInputFile::StreamerInputFile(const std::vector<std::string>& names):
  useIndex_(false),
  startMsg_(0),
  currentEvMsg_(0),
  headerBuf_(1000*1000),
  eventBuf_(1000*1000*7),
  currentFile_(0),
  streamerNames_(names),
  multiStreams_(true),
  currRun_(0),
  currProto_(0),
  newHeader_(false),
  endOfFile_(false)
{
  openStreamerFile(names.at(0));
  ++currentFile_;
  readStartMessage();
  currRun_ = startMsg_->run();
  currProto_ = startMsg_->protocolVersion();
}

void
StreamerInputFile::openStreamerFile(const std::string& name) {

  if (storage_) storage_->close();

  seal::IOOffset size = -1;
  if (StorageFactory::get()->check(name.c_str(), &size)) {
    try {
      storage_.reset(StorageFactory::get()->open(name.c_str(),
                                                 seal::IOFlags::OpenRead));
    }
    catch (seal::Error& se ) {
      throw cms::Exception("StreamerInputFile","StreamerInputFile")
        << "Error Opening Streamer Input File: " << name << "\n"
        << se.explain() << "\n";
    }
  }
  else {
    throw cms::Exception("StreamerInputFile", "StreamerInputFile")
      << "Error Opening Streamer Input File, file does not exist: "
      << name << "\n";
  }
}

const StreamerInputIndexFile* StreamerInputFile::index() {
  return index_;
}

seal::IOSize StreamerInputFile::readBytes(char *buf, seal::IOSize nBytes)
{
  seal::IOSize n;
  try {
    n = storage_->read(buf, nBytes);
  }
  catch (seal::Error& ce) {
    throw cms::Exception("StreamerInputFile","StreamerInputFile")
      << "Failed reading streamer file in function readBytes\n"
      << ce.explain() << "\n";
  }
  return n;
}

void StreamerInputFile::readStartMessage() 
{
  seal::IOSize nWant = sizeof(HeaderView);
  seal::IOSize nGot = readBytes(&headerBuf_[0], nWant);
  if (nGot != nWant) {
    throw cms::Exception("readStartMessage", "StreamerInputFile")
      << "Failed reading streamer file, first read in readStartMessage\n";
  }

  HeaderView head(&headerBuf_[0]);
  uint32 code = head.code();
  if (code != Header::INIT) /** Not an init message should return ******/
  {
    throw cms::Exception("readStartMessage", "StreamerInputFile")
      << "Expecting an init Message at start of file\n";
    return;
  }

  uint32 headerSize = head.size();
  if (headerBuf_.size() < headerSize) headerBuf_.resize(headerSize);

  if (headerSize > sizeof(HeaderView)) {
    nWant = headerSize - sizeof(HeaderView);
    nGot = readBytes(&headerBuf_[sizeof(HeaderView)], nWant);
    if (nGot != nWant) {
      throw cms::Exception("readStartMessage","StreamerInputFile")
        << "Failed reading streamer file, second read in readStartMessage\n";
    }
  }
  else {
    throw cms::Exception("readStartMessage","StreamerInputFile")
      << "Failed reading streamer file, init header size from data too small\n";
  }
  
  delete startMsg_;
  startMsg_ = new InitMsgView(&headerBuf_[0]) ;
}

bool StreamerInputFile::next()  
{
  if (useIndex_) {

     /** Read the offset of next event from Event Index */

     if (indexIter_b != indexIter_e) {
        EventIndexRecord* iview = *(indexIter_b);

        try {
          storage_->position((iview->getOffset()) - 1);
        }
        catch (seal::Error& ce) {
          throw cms::Exception("StreamerInputFile","StreamerInputFile")
            << "Failed reading streamer file in function next\n"
            << ce.explain() << "\n";
        }

        ++indexIter_b;
     }  
  }
  if (this->readEventMessage()) {
       return true;
  }

  if (multiStreams_) {
     //Try opening next file
     if (openNextFile()) {
        endOfFile_ = false;
        if (this->readEventMessage()) {
           return true;
        }
     }
  }

  return false;
}

bool StreamerInputFile::openNextFile() {

   if (currentFile_ <= streamerNames_.size()-1)
   {

     FDEBUG(10) << "Opening file "
                << streamerNames_.at(currentFile_).c_str() << std::endl;
 
     openStreamerFile(streamerNames_.at(currentFile_));

     // If start message was already there, then compare the
     // previous and new headers
     if (startMsg_ != NULL) {  
        FDEBUG(10) << "Comparing Header" << std::endl;
        if (!compareHeader())
        {
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
  if (currRun_ != startMsg_->run() ||
      currProto_ != startMsg_->protocolVersion())
  {
    throw cms::Exception("MismatchedInput","StreamerInputFile::compareHeader")
      << "File " << streamerNames_.at(currentFile_)
      << "\nhas different run number or protocol version than previous\n";

    return false;
  }
  newHeader_ = true;
  return true;
}


int StreamerInputFile::readEventMessage()  
{
  if (endOfFile_) return 0;

  seal::IOSize nWant = sizeof(HeaderView);
  seal::IOSize nGot = readBytes(&eventBuf_[0], nWant);
  if (nGot != nWant) {
    throw cms::Exception("readEventMessage", "StreamerInputFile")
      << "Failed reading streamer file, first read in readEventMessage\n";
  }

  HeaderView head(&eventBuf_[0]);
  uint32 code = head.code();

  // When we get the EOF record we know we have read all events
  // normally and are at the end, return 0 to indicate this
  if (code == Header::EOFRECORD) {
    endOfFile_ = true;
    return 0;
  }

  // If it is not an event nor EOFRECORD then something is wrong.
  if (code != Header::EVENT) {
    throw cms::Exception("readEventMessage", "StreamerInputFile")
      << "Failed reading streamer file, unknown code in event header\n"
      << "code = " << code << "\n";
  }

  uint32 eventSize = head.size();
  if (eventBuf_.size() < eventSize) eventBuf_.resize(eventSize);

  if (eventSize > sizeof(HeaderView)) {
    nWant = eventSize - sizeof(HeaderView);
    nGot = readBytes(&eventBuf_[sizeof(HeaderView)], nWant);
    if (nGot != nWant) {
      throw cms::Exception("readEventMessage", "StreamerInputFile")
        << "Failed reading streamer file, second read in readEventMessage\n";
    }
  }
  else {
    throw cms::Exception("readEventMessage","StreamerInputFile")
      << "Failed reading streamer file, event header size from data too small\n";
  }
 
  delete currentEvMsg_;
  currentEvMsg_ = new EventMsgView((void*)&eventBuf_[0]);
  
  return 1;
}
