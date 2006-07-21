#include "IOPool/Streamer/interface/StreamerFileIO.h"

StreamerInputIndexFile::~StreamerInputIndexFile()
{
  //ist_->close();
  //delete ist_;
  
  if (!startMsg_.init) {
    delete startMsg_.init;
    delete startMsg_.magic;
    delete startMsg_.reserved;
  }

  indexRecIter it;
  for(it = this->begin(); it != this->end(); ++it) 
     {
          delete (*it).eview;
          delete (*it).offset;
     }   
}

StreamerInputIndexFile::StreamerInputIndexFile(const string& name):
  ist_(new ifstream(name.c_str(), ios_base::binary | ios_base::in)),
  eof_(false),
  eventBufPtr_(0),
  headerBuf_(1000*1000),
  eventBuf_(1000*1000*10)
{
  readStartMessage();
  set_hlt_l1_sizes();
  while (readEventMessage()) {
  ;
  }
  ist_->close();
  delete ist_;
}


StreamerInputIndexFile::StreamerInputIndexFile(const vector<string>& names):
  eof_(false),
  eventBufPtr_(0),
  headerBuf_(1000*1000),
  eventBuf_(1000*1000*10)
{
   for (unsigned int i=0; i!=names.size(); ++i) 
   {
       ist_ = new ifstream(names.at(i).c_str(), ios_base::binary | ios_base::in);


       readStartMessage();
       set_hlt_l1_sizes();
       while (readEventMessage()) {
       ;
       }
       ist_->close();
       delete ist_; 
   }
}

void StreamerInputIndexFile::readStartMessage() {
  ist_->clear();
  ist_->read((char*)&headerBuf_[0], headerBuf_.size());
  startMsg_.magic = new uint32 (convert32((unsigned char*)&headerBuf_[0]));
  startMsg_.reserved = (uint64*)new long long 
		(convert64((unsigned char*)&headerBuf_[4]));
  startMsg_.init = new InitMsgView(&headerBuf_[12]);
  uint32 headerSize = startMsg_.init->headerSize();
  //Bring the ist_ at end of Header
  ist_->clear();
  ist_->seekg(sizeof(uint32)+sizeof(uint64)+headerSize, ios::beg);
  headerBuf_.resize(headerSize);
}

int StreamerInputIndexFile::readEventMessage()  {
  int last_pos = ist_->tellg();


  ist_->clear();
  ist_->read((char*)&eventBuf_[eventBufPtr_+1], 5);

  if (ist_->gcount() < 1) {
	eof_ = true;
	return 0;
  }
 
  HeaderView head_(&eventBuf_[eventBufPtr_+1]);
  uint32 code = head_.code();
  if (code != Header::EVENT) /** Not an event message should return */
     return 0;

  uint32 eventSize =  head_.size();
  //Bring the pointer to end of Start Message
  ist_->clear();
  ist_->seekg(last_pos, ios::beg);
  
  // Read more than Header, Later we can have a 
  // HeaderSize Field in the Message
  ist_->read((char*)&eventBuf_[eventBufPtr_+1], eventSize);
  if (ist_->gcount() < 1) {
     eof_= true;
     return 0;
  }

  EventIndexRecord currentEvMsg_;
  currentEvMsg_.eview = new EventMsgView((void*)&eventBuf_[eventBufPtr_+1], 
					 hlt_bit_cnt_, l1_bit_cnt_);
  currentEvMsg_.offset = (uint64*) new long long (convert64((unsigned char*)
				   &eventBuf_[eventBufPtr_+1+currentEvMsg_.eview->
					      headerSize()]));

  indexes_.push_back(currentEvMsg_);

  //This Brings the pointer to end of this Event Msg.
  uint32 new_len = currentEvMsg_.eview->headerSize() + sizeof(uint64);
  //eventBuf_.resize(new_len);
  ist_->clear();
  ist_->seekg(last_pos+new_len, ios::beg);

  eventBufPtr_ +=  new_len;
  return 1;
}

bool header_event_sorter(EventIndexRecord first, EventIndexRecord second) {
    //uint32 event_first = first.eview->event(); 
    //uint32 event_second = second.eview->event();
     
    if( first.eview->event() > second.eview->event() ) 
	return true;
    return false;
}

bool header_run_sorter(EventIndexRecord first, EventIndexRecord second) {
    //uint32 run_first = first.eview->run();
    //uint32 run_second = second.eview->run();
     
    if( first.eview->run() > second.eview->run() )
        return true;
    return false;
}


indexRecIter StreamerInputIndexFile::sort() {
  //Run sorting is required ?? 
  std::sort( this->begin(), this->end(), header_run_sorter);
  std::sort( this->begin(), this->end(), header_event_sorter);
  return this->begin();
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

