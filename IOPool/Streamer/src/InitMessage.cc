#include "IOPool/Streamer/interface/InitMessage.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
#include <iterator>

InitMsgView::InitMsgView(void* buf):
  buf_((uint8*)buf),head_(buf)
{
  if (protocolVersion() == 2) {
      std::cout << "Protocol Version 2 encountered" << std::endl; 
      release_start_ = buf_ + sizeof(InitHeader) - (sizeof(uint32)*2);
      // Minus the size for Init and Event Header size fileds
      // in the InitHeader
  } else { //For version 3 
      release_start_ = buf_ + sizeof(InitHeader);
  }
  release_len_ = *release_start_;
  release_start_ += sizeof(uint8);

  //Lets get Process Name from right after Release Name  
  if (protocolVersion() > 3) {
	std::cout << "Protocol Version > 3 encountered" << std::endl;
	processName_len_ = *(release_start_ + release_len_);
	processName_start_ = (uint8*)(release_start_ + release_len_ + sizeof(uint8));

  	hlt_trig_start_ = processName_start_ + processName_len_;

  } else {
  	hlt_trig_start_ = release_start_ + release_len_;
  }

  hlt_trig_count_ = convert32(hlt_trig_start_);
  hlt_trig_start_ += sizeof(char_uint32);
  hlt_trig_len_ = convert32(hlt_trig_start_);
  hlt_trig_start_ += sizeof(char_uint32);
  l1_trig_start_ = hlt_trig_start_ + hlt_trig_len_;
  l1_trig_count_ = convert32(l1_trig_start_);
  l1_trig_start_ += sizeof(char_uint32);
  l1_trig_len_ = convert32(l1_trig_start_);
  l1_trig_start_ += sizeof(char_uint32);
  desc_start_ = l1_trig_start_ + l1_trig_len_;
  desc_len_ = convert32(desc_start_);
  desc_start_ += sizeof(char_uint32);
}

uint32 InitMsgView::run() const
{
  InitHeader* h = reinterpret_cast<InitHeader*>(buf_);
  return convert32(h->run_);
}

uint32 InitMsgView::protocolVersion() const
{
  InitHeader* h = reinterpret_cast<InitHeader*>(buf_);
  return h->version_.protocol_;
}

void InitMsgView::pset(uint8* put_here) const
{
  InitHeader* h = reinterpret_cast<InitHeader*>(buf_);
  memcpy(put_here,h->version_.pset_id_,sizeof(h->version_.pset_id_));
}

std::string InitMsgView::releaseTag() const
{
  return std::string(reinterpret_cast<char *>(release_start_),release_len_);
}

std::string InitMsgView::processName() const
{
   if (protocolVersion() < 4)
      throw cms::Exception("Invalid Message Version", "InitMsgView")
        << "Process Name is only supported in Protocol Version 4 and above" << ".\n";

   return std::string(reinterpret_cast<char *>(processName_start_),processName_len_);
}


void InitMsgView::hltTriggerNames(Strings& save_here) const
{
  getNames(hlt_trig_start_,hlt_trig_len_,save_here);
}

void InitMsgView::l1TriggerNames(Strings& save_here) const
{
  getNames(l1_trig_start_,l1_trig_len_,save_here);
}

void InitMsgView::getNames(uint8* from, uint32 from_len, Strings& to) const
{
  // not the most efficient way to do this
  std::istringstream ist(std::string(reinterpret_cast<char *>(from),from_len));
  typedef std::istream_iterator<std::string> Iter;
  std::copy(Iter(ist),Iter(),std::back_inserter(to));
}

uint32 InitMsgView::eventHeaderSize() const
{
  if (protocolVersion() == 2) {
       /** This is estimated size of event header for Protocol Version 2. */

       uint32 hlt_sz = get_hlt_bit_cnt();
       if (hlt_sz != 0 ) hlt_sz = 1+ ((hlt_sz-1)/4);

       uint32 l1_sz = get_l1_bit_cnt();
       if (l1_sz != 0) l1_sz = 1 + ((l1_sz-1)/8);

       return 1 + (4*8) + hlt_sz+l1_sz; 
   }

   InitHeader* h = reinterpret_cast<InitHeader*>(buf_);
   return convert32(h->event_header_size_);
}

/***
uint32 InitMsgView::initHeaderSize() const
{
  InitHeader* h = reinterpret_cast<InitHeader*>(buf_);
  return convert32(h->init_header_size_);
} **/


