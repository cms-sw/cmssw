#include "IOPool/Streamer/interface/StreamerFileIO.h"

StreamerInputFile::~StreamerInputFile()
{
  ist_->close();
  delete ist_;
  
  if (!startMsg_) {
    delete startMsg_;
  }
  
  if (!currentEvMsg_ ) {
    delete  currentEvMsg_;
  }
  
}

StreamerInputFile::StreamerInputFile(const string& name):
  ist_(new ifstream(name.c_str(), ios_base::binary | ios_base::in)),
  useIndex_(false),
  headerBuf_(1000*1000),
  eventBuf_(1000*1000*7)
{
  readStartMessage();
  set_hlt_l1_sizes();
}

StreamerInputFile::StreamerInputFile(const string& name, 
                                     const string& order):
  ist_(new ifstream(name.c_str(), ios_base::binary | ios_base::in)),
  useIndex_(true),
  index_(new StreamerInputIndexFile(order)),
  //indexIter_b(index_->begin()),
  indexIter_b(index_->sort()),
  indexIter_e(index_->end()),
  headerBuf_(1000*1000),
  eventBuf_(1000*1000*7)
{
  readStartMessage();
  set_hlt_l1_sizes();
}

StreamerInputFile::StreamerInputFile(const string& name,
                                     const StreamerInputIndexFile& order):
  ist_(new ifstream(name.c_str(), ios_base::binary | ios_base::in)),
  useIndex_(true),
  index_((StreamerInputIndexFile*)&order),
  //indexIter_b(index_->begin()),
  indexIter_b(index_->sort()),
  indexIter_e(index_->end()),
  headerBuf_(1000*1000),
  eventBuf_(1000*1000*7)
{
  readStartMessage();
  set_hlt_l1_sizes();
}


/**
StreamerInputFile::StreamerInputFile(const vector<string>& names)
{




}
**/


StreamerInputIndexFile* StreamerInputFile::index() {
  return index_;
}

void StreamerInputFile::readStartMessage() 
{
  ist_->read((char*)&headerBuf_[0], 5);
  uint32 headerSize = convert32((unsigned char*)&headerBuf_[1]);
  //Bring the pointer start of Start Message/ start of file
  ist_->seekg(0, ios::beg);
  ist_->read((char*)&headerBuf_[0], headerSize);
  
  startMsg_ = new InitMsgView(&headerBuf_[0]) ;
  headerBuf_.resize(startMsg_->size());
}

bool StreamerInputFile::next()  
{
  if (useIndex_) {
     /** Read the offset of next event from Event Index */

     if ( indexIter_b != indexIter_e ) 
        {
        EventIndexRecord* iview = (EventIndexRecord*) &(*indexIter_b);
        ist_->clear();
        // Bring the fptr to start of event 
        ist_->seekg( (*(iview->offset)) - 1, ios::beg);
        indexIter_b++;
      }  
  }
  if ( this->readEventMessage() ) {
       return true;
  }
  return false;
}

int StreamerInputFile::readEventMessage()  
{  
  ist_->clear();
  int last_pos = ist_->tellg();
  ist_->read((char*)&eventBuf_[0], 5);
  if (ist_->gcount() < 1)
    return 0;

  HeaderView head_(&eventBuf_[0]);
  uint32 code = head_.code();
  if (code != Header::EVENT) /** Not an event message should return ******/
    return 0;

  uint32 eventSize =  head_.size();
  //Bring the pointer to end of Start Message
  ist_->clear();
  ist_->seekg(last_pos, ios::beg);
  ist_->read((char*)&eventBuf_[0], eventSize);
  if (ist_->gcount() < 1) return 0;
  
  currentEvMsg_ = new EventMsgView((void*)&eventBuf_[0], 
				   hlt_bit_cnt_, l1_bit_cnt_) ;
  
  //This Brings the pointer to end of this Event Msg.
  // HWKC should not be resizing eventBuf as its assumed to be the max size later!
  //eventBuf_.resize(currentEvMsg_->size());
  ist_->clear();
  ist_->seekg(last_pos+currentEvMsg_->size());

  return 1;
}

void StreamerInputFile::set_hlt_l1_sizes() 
{  
  Strings vhltnames,vl1names;
  startMsg_->hltTriggerNames(vhltnames);
  startMsg_->l1TriggerNames(vl1names);
  hlt_bit_cnt_ = vhltnames.size();
  l1_bit_cnt_ = vl1names.size(); 
}

uint32 StreamerInputFile::get_hlt_bit_cnt() 
{
 return hlt_bit_cnt_;
}


uint32 StreamerInputFile::get_l1_bit_cnt()
{
 return l1_bit_cnt_;
}


