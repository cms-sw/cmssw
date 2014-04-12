#include "IOPool/Streamer/interface/InitMessage.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
#include <iterator>
#include <cstring>

InitMsgView::InitMsgView(void* buf) :
  buf_((uint8*)buf),
  head_(buf),
  release_start_(0),
  release_len_(0),
  processName_start_(0),
  processName_len_(0),
  outputModuleLabel_start_(0),
  outputModuleLabel_len_(0),
  outputModuleId_(0),
  hlt_trig_start_(0),
  hlt_trig_count_(0),
  hlt_trig_len_(0),
  hlt_select_start_(0),
  hlt_select_count_(0),
  hlt_select_len_(0),
  l1_trig_start_(0),
  l1_trig_count_(0),
  l1_trig_len_(0),
  adler32_chksum_(0),
  host_name_start_(0),
  host_name_len_(0),
  desc_start_(0),
  desc_len_(0) {
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
  uint8* pos = release_start_ + release_len_;

  //Lets get Process Name from right after Release Name
  if (protocolVersion() > 3) {
        //std::cout << "Protocol Version > 3 encountered" << std::endl;
        processName_len_ = *pos;
        processName_start_ = (uint8*)(pos + sizeof(uint8));
        pos = processName_start_ + processName_len_;

        // Output Module Label
        if (protocolVersion() > 4) {
            outputModuleLabel_len_ = *pos;
            outputModuleLabel_start_ = (uint8*)(pos + sizeof(uint8));
            pos = outputModuleLabel_start_ + outputModuleLabel_len_;

            // Output Module Id
            if (protocolVersion() > 5) {
              outputModuleId_ = convert32(pos);
              pos += sizeof(char_uint32);
            }
        }
  }

  hlt_trig_start_ = pos;
  hlt_trig_count_ = convert32(hlt_trig_start_);
  hlt_trig_start_ += sizeof(char_uint32);
  hlt_trig_len_ = convert32(hlt_trig_start_);
  hlt_trig_start_ += sizeof(char_uint32);
  pos = hlt_trig_start_ + hlt_trig_len_;

  if (protocolVersion() > 4) {
      hlt_select_start_ = pos;
      hlt_select_count_ = convert32(hlt_select_start_);
      hlt_select_start_ += sizeof(char_uint32);
      hlt_select_len_ = convert32(hlt_select_start_);
      hlt_select_start_ += sizeof(char_uint32);
      pos = hlt_select_start_ + hlt_select_len_;
  }

  l1_trig_start_ = pos;
  l1_trig_count_ = convert32(l1_trig_start_);
  l1_trig_start_ += sizeof(char_uint32);
  l1_trig_len_ = convert32(l1_trig_start_);
  l1_trig_start_ += sizeof(char_uint32);
  pos = l1_trig_start_ + l1_trig_len_;

  if (protocolVersion() > 7) {
    adler32_chksum_ = convert32(pos);
    pos += sizeof(uint32);

    if (protocolVersion() <= 9) {
      host_name_start_ = pos;
      host_name_len_ = *host_name_start_;
      host_name_start_ += sizeof(uint8);
      pos = host_name_start_ + host_name_len_;
    }
  }

  desc_start_ = pos;
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

std::string InitMsgView::outputModuleLabel() const
{
   if (protocolVersion() < 5)
      throw cms::Exception("Invalid Message Version", "InitMsgView")
        << "Output Module Label is only supported in Protocol Version 5 and above" << ".\n";

   return std::string(reinterpret_cast<char *>(outputModuleLabel_start_),outputModuleLabel_len_);
}


void InitMsgView::hltTriggerNames(Strings& save_here) const
{
  MsgTools::getNames(hlt_trig_start_,hlt_trig_len_,save_here);
}

void InitMsgView::hltTriggerSelections(Strings& save_here) const
{
  if (protocolVersion() < 5)
    throw cms::Exception("Invalid Message Version", "InitMsgView")
      << "HLT trigger selections are only supported in Protocol Version 5 and above" << ".\n";

  MsgTools::getNames(hlt_select_start_,hlt_select_len_,save_here);
}

void InitMsgView::l1TriggerNames(Strings& save_here) const
{
  MsgTools::getNames(l1_trig_start_,l1_trig_len_,save_here);
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

std::string InitMsgView::hostName() const
{
  if (host_name_start_) {
    std::string host_name(reinterpret_cast<char *>(host_name_start_),host_name_len_);
    size_t found = host_name.find('\0');
    if(found != std::string::npos) {
      return std::string(host_name, 0, found);
    } else {
      return host_name;
    }
  }
  else {
    return "n/a";
  }
}
