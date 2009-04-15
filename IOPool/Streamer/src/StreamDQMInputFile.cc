#include "IOPool/Streamer/interface/StreamDQMInputFile.h"
#include "IOPool/Streamer/interface/MsgTools.h"
#include "FWCore/Utilities/interface/Exception.h"
//#include "FWCore/Utilities/interface/DebugMacros.h"

using namespace edm;

StreamDQMInputFile::~StreamDQMInputFile()
{
  //ist_->close();
}

StreamDQMInputFile::StreamDQMInputFile(const std::string& name):
  currentEvMsg_(),
  ist_(new ifstream(name.c_str())),
  eventBuf_(1000*1000*7)
{

  if (!ist_->is_open()) {
      throw cms::Exception("StreamDQMInputFile","StreamDQMInputFile")
        << "Error Opening Input File: " << name << "\n";
  }

}

bool StreamDQMInputFile::next()  
{
  if (this->readDQMEventMessage()) {
       return true;
  }

  return false;
}

int StreamDQMInputFile::readDQMEventMessage() {

  uint32 last_pos = ist_->tellg();
  uint32 nWant = sizeof(HeaderView);

  ist_->read(&eventBuf_[0], nWant);

  if (ist_->eof() || static_cast<unsigned int>(ist_->gcount()) < nWant) {
        return 0;
  }  

  HeaderView head(&eventBuf_[0]);
  uint32 code = head.code();
  if (code != Header::DQM_EVENT)
    return 0;

  //This includes header
  uint32 eventSize = head.size();
  if (eventBuf_.size() < eventSize) eventBuf_.resize(eventSize);

  if (eventSize > sizeof(DQMEventMsgView)) {
    //Lets read the whole thing again
    nWant = eventSize;
    ist_->seekg(last_pos, std::ios::beg);
    ist_->read(&eventBuf_[0], nWant);
    if (ist_->eof() || static_cast<unsigned int>(ist_->gcount()) < nWant) {
        return 0;
    }
  }

  currentEvMsg_.reset(new DQMEventMsgView((void*)&eventBuf_[0]));
  uint32 new_len = last_pos + eventSize; 
  ist_->seekg(new_len, std::ios::beg);
  return 1;
}

