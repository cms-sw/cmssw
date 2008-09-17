#include "IOPool/Streamer/interface/StreamerInputIndexFile.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

#include<fstream>
#include<iostream>

using namespace edm;

StreamerInputIndexFile::~StreamerInputIndexFile()
{
  delete startMsg_; 

  for(indexRecIter it = this->begin(), itEnd = this->end(); it != itEnd; ++it) {
          delete (*it);
  }
}

StreamerInputIndexFile::StreamerInputIndexFile(const std::string& name):
  ist_(new std::ifstream(name.c_str(), std::ios_base::binary | std::ios_base::in)),
  startMsg_(0),
  eof_(false),
  eventBufPtr_(0),
  headerBuf_(1000*1000),
  eventBuf_(1000*1000*40),
  eventHeaderSize_(0),
  indexes_(0)
{

  FDEBUG(10) << "Opening Index file" << std::endl;
  if (!ist_->is_open()) {
       throw cms::Exception ("StreamerInputIndexFile","StreamerInputIndexFile")
          << "Error Opening Input File: "<< name<< "\n";
  } 
  readStartMessage();
  while (readEventMessage()) {
      ;
  }
  
  ist_->close();
  delete ist_;
}


StreamerInputIndexFile::StreamerInputIndexFile(const std::vector<std::string>& names):
  startMsg_(0),
  eof_(false),
  eventBufPtr_(0),
  headerBuf_(1000*1000),
  eventBuf_(1000*1000*40),
  eventHeaderSize_(0),
  indexes_(0)
{
   for (unsigned int i=0; i!=names.size(); ++i) 
   {
     ist_ = new std::ifstream(names.at(i).c_str(), std::ios_base::binary | std::ios_base::in);
     if (!ist_->is_open())
     {
       throw cms::Exception ("StreamerInputIndexFile","StreamerInputIndexFile")
          << "Error Opening Input File: "<< names.at(i) << "\n";
     }

       readStartMessage();
       while (readEventMessage()) {
       ;
       }
       ist_->close();
       delete ist_; 
   }
}

void StreamerInputIndexFile::readStartMessage() {
  //Read magic+reserved fileds at the start of file 
  //ist_->clear();
  ist_->read((char*)&headerBuf_[0], sizeof(StartIndexRecordHeader));
  if (ist_->eof() || static_cast<unsigned int>(ist_->gcount()) < sizeof(StartIndexRecordHeader)) {
        return;
   }  
  //Read Header from the start of init message to find the size
  ist_->read((char*)&headerBuf_[sizeof(StartIndexRecordHeader)], sizeof(HeaderView));
  if (ist_->eof() || static_cast<unsigned int>(ist_->gcount()) < sizeof(HeaderView))
  {
        throw cms::Exception("readStartMessage","StreamerInputFile")
              << "Empty file encountered\n";
  }

  HeaderView head(&headerBuf_[sizeof(StartIndexRecordHeader)]);
  uint32 code = head.code();
  if (code != Header::INIT) // ** Not an init message should return ****** /
  {
    throw cms::Exception("readStartMessage","StreamerInputFile")
              << "Expecting an init Message at start of file\n";
    return;
  }
  uint32 headerSize = head.size();
  //Bring the pointer at start of Start Message (Starts after file header magic+reserved)
  ist_->seekg(sizeof(StartIndexRecordHeader), std::ios::beg);
  if (headerBuf_.size() < (sizeof(StartIndexRecordHeader) + headerSize))
    headerBuf_.resize(sizeof(StartIndexRecordHeader) + headerSize);
  ist_->read((char*)&headerBuf_[sizeof(StartIndexRecordHeader)], headerSize);
 
  delete startMsg_;
   
  startMsg_ = new StartIndexRecord();
  startMsg_->makeHeader(&headerBuf_[0]);
  //Init msg lies just after StartIndexRecordHeader
  startMsg_->makeInit(&headerBuf_[sizeof(StartIndexRecordHeader)]);
   
  //   As the size of index record is unknown as of yet the 
  //   ist_ can over run, so it may have reached EOF
  //   it needs to be reset, before its positioned correctly.

  ist_->clear();
  //Bring ist_ at the end of record
  headerSize = (startMsg_->getInit())->headerSize(); 
  ist_->seekg(sizeof(StartIndexRecordHeader)+headerSize, std::ios::beg);

  eventHeaderSize_ = (startMsg_->getInit())->eventHeaderSize();
}

int StreamerInputIndexFile::readEventMessage()  {
  std::streampos last_pos = ist_->tellg();
  uint32 bufPtr = eventBufPtr_;
  //uint32 bufPtr = eventBufPtr_+1;

  //ist_->clear();
  if (eventBuf_.size() < bufPtr + sizeof(HeaderView)) {
    throw cms::Exception("readEventMessage","StreamerInputFile")
      << "eventBuf array is about to overflow, just before first read.\n";
  }
  ist_->read((char*)&eventBuf_[bufPtr], sizeof(HeaderView));

  if (ist_->eof() || static_cast<unsigned int>(ist_->gcount()) < sizeof(HeaderView))
  {      
	eof_ = true;
	return 0;
  }

  HeaderView head_(&eventBuf_[bufPtr]);
  uint32 code = head_.code();

  if (code != Header::EVENT) /** Not an event message should return */
     {
     FDEBUG(10) << "Not an event Message "<< std::endl;
     return 0;
     }

  //Bring the pointer at last position, start of event msg
  //ist_->clear();
  ist_->seekg(last_pos);
  
  if (eventBuf_.size() < bufPtr + eventHeaderSize_ + sizeof(uint64)) {
    throw cms::Exception("readEventMessage","StreamerInputFile")
      << "eventBuf array is about to overflow, just before second read.\n";
  }
  ist_->read((char*)&eventBuf_[bufPtr], eventHeaderSize_+sizeof(uint64));
  if (ist_->eof()) {
     eof_= true;
     return 0;
  }

  EventIndexRecord* currentEvMsg = new EventIndexRecord();
  //EventIndexRecord currentEvMsg;
  currentEvMsg->makeEvent((void*)&eventBuf_[bufPtr]);

  uint32 offset_loc = bufPtr + eventHeaderSize_;
   
  currentEvMsg->makeOffset(&eventBuf_[offset_loc]);

  indexes_.push_back(currentEvMsg);

  //This Brings the pointer to end of this Event Msg.
  std::streamoff new_len = eventHeaderSize_ + sizeof(uint64); 

  //How many bytes have read so far
  eventBufPtr_ +=  eventHeaderSize_ + sizeof(uint64);

  //This should be the proper position of the file pointer
  ist_->seekg(last_pos + new_len);
  return 1;
}

bool header_event_sorter(EventIndexRecord* first, EventIndexRecord* second) {
    //uint32 event_first = first.eview->event(); 
    //uint32 event_second = second.eview->event();
     
    if(((first->getEventView())->event()) > ((second->getEventView())->event())) 
	return true;
    return false;
}

bool header_run_sorter(EventIndexRecord* first, EventIndexRecord* second) {
    //uint32 run_first = first.eview->run();
    //uint32 run_second = second.eview->run();
     
    if(((first->getEventView())->run()) > ((second->getEventView())->run()))
        return true;
    return false;
}

indexRecIter StreamerInputIndexFile::sort() {
  //Run sorting is required ?? 
  sort_all(*this, header_run_sorter);
  sort_all(*this, header_event_sorter);
  return this->begin();
}
