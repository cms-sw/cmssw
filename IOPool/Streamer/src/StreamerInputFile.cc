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
  filename_(name),
  ist_(new ifstream(name.c_str(), ios_base::binary | ios_base::in)),
  //ist_(makeInputFile(filename_)),
  headerBuf_(1000*1000),
  eventBuf_(1000*1000*7)
{
  readStartMessage();
  set_hlt_l1_sizes();
}

void StreamerInputFile::readStartMessage() {
  
  ist_->read((char*)&headerBuf_[0], 5);
  uint32 headerSize = convert32((unsigned char*)&headerBuf_[1]);
  //Bring the pointer start of Start Message/ start of file
  ist_->seekg(0, ios::beg);
  ist_->read((char*)&headerBuf_[0], headerSize);
  
  startMsg_ = new InitMsgView(&headerBuf_[0], ist_->gcount() ) ;
  headerBuf_.resize(startMsg_->size());
}

bool StreamerInputFile::next()  
{
  if ( this->readEventMessage() )
    {
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
  
  uint32 eventSize = convert32((unsigned char*)&eventBuf_[1]);
  //Bring the pointer to end of Start Message
  ist_->clear();
  ist_->seekg(last_pos, ios::beg);
  ist_->read((char*)&eventBuf_[0], eventSize);
  if (ist_->gcount() < 1) return 0;
  
  cout<<"hlt_bit_cnt_:::::::::"<<hlt_bit_cnt_<<endl;
  cout<<"l1_bit_cnt_:::::::::::"<<l1_bit_cnt_<<endl; 

  currentEvMsg_ = new EventMsgView((void*)&eventBuf_[0], 
				   (uint32)ist_->gcount(),
				   hlt_bit_cnt_, l1_bit_cnt_) ;
  
  //This Brings the pointer to end of this Event Msg.
  eventBuf_.resize(currentEvMsg_->size());
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


