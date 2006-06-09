#include "IOPool/Streamer/interface/StreamerFileIO.h"

StreamerInputIndexFile::~StreamerInputIndexFile()
{
  ist_->close();
  delete ist_;
  
  if (!startMsg_.init) {
    delete startMsg_.init;
  }
  
  if (!currentEvMsg_.eview) {
    delete  currentEvMsg_.eview;
  }
  
}

StreamerInputIndexFile::StreamerInputIndexFile(const string& name):
  filename_(name),
  ist_(new ifstream(name.c_str(), ios_base::binary | ios_base::in)),
  //ist_(makeInputFile(filename_)),
  headerBuf_(1000*1000),
  eventBuf_(1000*1000*7)
{
  readStartMessage();
  set_hlt_l1_sizes();
}

void StreamerInputIndexFile::readStartMessage() {
  if (!*ist_) { cout <<"Error"<<endl; }
  
  ist_->clear();
  ist_->read((char*)&headerBuf_[0], headerBuf_.size());
  startMsg_.magic = convert32((unsigned char*)&headerBuf_[0]);
  startMsg_.reserved = convert64((unsigned char*)&headerBuf_[4]);
  startMsg_.init = new InitMsgView(&headerBuf_[12], ist_->gcount() );
  uint32 headerSize = startMsg_.init->headerSize();
  //Bring the ist_ at end of Header
  ist_->clear();
  ist_->seekg(sizeof(uint32)+sizeof(uint64)+headerSize, ios::beg);
  headerBuf_.resize(headerSize);
}

bool StreamerInputIndexFile::next()  {
  if ( this->readEventMessage() )
    {
      return true;
    }
  return false;
  
}

int StreamerInputIndexFile::readEventMessage()  {
  int last_pos = ist_->tellg();
  ist_->clear();
  ist_->read((char*)&eventBuf_[0], 5);
  if (ist_->gcount() < 1) return 0;
  
  uint32 eventSize = convert32((unsigned char*)&eventBuf_[1]);
  //Bring the pointer to end of Start Message
  ist_->clear();
  ist_->seekg(last_pos, ios::beg);
  
  // Read more than Header, Later we can have a 
  // HeaderSize Field in the Message
  ist_->read((char*)&eventBuf_[0], eventSize);
  if (ist_->gcount() < 1) return 0;

  cout<<"hlt_bit_cnt_::::I:::::"<<hlt_bit_cnt_<<endl;
  cout<<"l1_bit_cnt_:::::I::::::"<<l1_bit_cnt_<<endl; 
  
  currentEvMsg_.eview = new EventMsgView((void*)&eventBuf_[0], 
					 (uint32)ist_->gcount(),
					 hlt_bit_cnt_, l1_bit_cnt_);
  currentEvMsg_.offset = convert64((unsigned char*)
				   &eventBuf_[currentEvMsg_.eview->
					      headerSize()]);
  //This Brings the pointer to end of this Event Msg.
  uint32 new_pos = currentEvMsg_.eview->headerSize() + sizeof(uint64);
  eventBuf_.resize(new_pos);
  ist_->clear();
  ist_->seekg(last_pos+new_pos, ios::beg);
  return 1;
  //}
  return 0;
}

void StreamerInputIndexFile::set_hlt_l1_sizes() {
  
  Strings vhltnames,vl1names;
  startMsg_.init->hltTriggerNames(vhltnames);
  startMsg_.init->l1TriggerNames(vl1names);
  hlt_bit_cnt_ = vhltnames.size();
  l1_bit_cnt_ = vl1names.size();
  
}

uint32 StreamerInputIndexFile::get_hlt_bit_cnt()
{
 return hlt_bit_cnt_;
}


uint32 StreamerInputIndexFile::get_l1_bit_cnt()
{
 return l1_bit_cnt_;
}

