#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/StreamerInputFile.h"
#include "IOPool/Streamer/interface/StreamerInputIndexFile.h"
#include "IOPool/Streamer/interface/StreamerFileIO.h"
#include "IOPool/Streamer/interface/Utilities.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

using namespace edm;

StreamerInputFile::~StreamerInputFile()
{

  if (ist_ != NULL) {
    ist_->close();
    delete ist_;
  }
  
  if (startMsg_ != NULL) {
    delete startMsg_;
  }
  
  if (currentEvMsg_ != NULL) {
    delete  currentEvMsg_;
  }
  
}

StreamerInputFile::StreamerInputFile(const string& name):
  ist_(new ifstream(name.c_str(), ios_base::binary | ios_base::in)),
  useIndex_(false),
  startMsg_(0),
  currentEvMsg_(0),
  headerBuf_(1000*1000),
  eventBuf_(1000*1000*7),
  multiStreams_(false),
  newHeader_(false)
{
  readStartMessage();
}

StreamerInputFile::StreamerInputFile(const string& name, 
                                     const string& order):
  ist_(new ifstream(name.c_str(), ios_base::binary | ios_base::in)),
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
  newHeader_(false)
{
  readStartMessage(); 
}

StreamerInputFile::StreamerInputFile(const string& name,
                                     const StreamerInputIndexFile& order):
  ist_(new ifstream(name.c_str(), ios_base::binary | ios_base::in)),
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
  newHeader_(false)
{
  readStartMessage(); 
}


StreamerInputFile::StreamerInputFile(const vector<string>& names):
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
 newHeader_(false)
{
  ist_ =new ifstream(names.at(0).c_str(), ios_base::binary | ios_base::in);

  currentFile_++;
  
  readStartMessage();
  
  currRun_ = startMsg_->run();
  currProto_ = startMsg_->protocolVersion();
}


const StreamerInputIndexFile* StreamerInputFile::index() {
  return index_;
}

void StreamerInputFile::readStartMessage() 
{
  ist_->read((char*)&headerBuf_[0], sizeof(HeaderView));

  if (ist_->eof() || (unsigned int)ist_->gcount() < sizeof(HeaderView)  ) 
  {
        throw cms::Exception("readStartMessage","StreamerInputFile")
              << "No file exists or Empty file encountered:\n";
  }

  HeaderView head(&headerBuf_[0]);
  uint32 code = head.code();
  if (code != Header::INIT) /** Not an init message should return ******/
  {
    throw cms::Exception("readStartMessage","StreamerInputFile")
              << "Expecting an init Message at start of file\n";
    return;
  }

  uint32 headerSize = head.size();
  //Bring the pointer at start of Start Message/start of file
  ist_->seekg(0, ios::beg);
  ist_->read((char*)&headerBuf_[0], headerSize);
 
  if (startMsg_ != NULL) 
  {
     delete startMsg_;
  }
  startMsg_ = new InitMsgView(&headerBuf_[0]) ;
}

bool StreamerInputFile::next()  
{
  if (useIndex_) {
     /** Read the offset of next event from Event Index */

     if ( indexIter_b != indexIter_e ) 
        {
        EventIndexRecord* iview = *(indexIter_b);
        //ist_->clear();
        // Bring the fptr to start of event 
        ist_->seekg( (iview->getOffset()) - 1, ios::beg);
        indexIter_b++;
      }  
  }
  if ( this->readEventMessage() ) {
       return true;
  }

  if (multiStreams_) {
     //Try opening next file
     if (openNextFile() ) {
        if ( this->readEventMessage() ) {
           return true;
        }
     }
  }

  return false;
}

bool StreamerInputFile::openNextFile() {

   if (currentFile_ <= streamerNames_.size()-1)
   {

     FDEBUG(10) << "Opening file "<< streamerNames_.at(currentFile_).c_str() << std::endl;
 
     if (ist_ != NULL ) {
       ist_->clear();
       ist_->close();
       delete ist_;
     }

     ist_ = new ifstream(streamerNames_.at(currentFile_).c_str(),
                                 ios_base::binary | ios_base::in);
     //if start message was already there, lets see if the new one is similar
     if (startMsg_ != NULL ) {  //There was a previous file opened, must compare headers
        FDEBUG(10) << "Comparing Header"<<endl;
        if ( !compareHeader() )
        {
            return false;
        }
     }
     currentFile_++;
     return true;
   }
   return false;
}

bool StreamerInputFile::compareHeader() {

  //Get the new header
  readStartMessage();
  
  //Values from new Header should match up
  if ( currRun_ != startMsg_->run() ||
       currProto_ != startMsg_->protocolVersion() )
     {
      throw cms::Exception("MismatchedInput","StreamerInputFile::compareHeader")
        << "File " << streamerNames_.at(currentFile_).c_str() 
        << "\nhas different run number or protocol version then previous\n";

      return false;
     }

  newHeader_ = true;
  return true;
}


int StreamerInputFile::readEventMessage()  
{  
  int last_pos = ist_->tellg();
  ist_->read((char*)&eventBuf_[0], sizeof(HeaderView));
  if (ist_->eof() || (unsigned int)ist_->gcount() < sizeof(HeaderView)  )
        return 0;

  HeaderView head(&eventBuf_[0]);
  uint32 code = head.code();
  if (code != Header::EVENT) /** Not an event message should return ******/
    return 0;

  uint32 eventSize =  head.size();
  //Bring the pointer to end of previous Message
  
  ist_->seekg(last_pos, ios::beg);
  ist_->read((char*)&eventBuf_[0], eventSize);
  if (ist_->eof() || (unsigned int)ist_->gcount() < sizeof(eventSize)  ) //Probably an unfinished file
     return 0;

  if (currentEvMsg_ != NULL) {
      delete currentEvMsg_;
  }
  
  currentEvMsg_ = new EventMsgView((void*)&eventBuf_[0]);
  
  //This Brings the pointer to end of this Event Msg.
  ist_->seekg(last_pos+currentEvMsg_->size());

  return 1;
}


