#include "EventFilter/CSCRawToDigi/interface/CSCDDUHeader.h"
#include <string.h> // for bzero
#include <iostream>

CSCDDUHeader::CSCDDUHeader() 
{
  bzero(this, sizeInWords()*2);
  init();
}


CSCDDUHeader::CSCDDUHeader(unsigned bx, unsigned l1num, unsigned sourceId, unsigned fmt_version) 
  : format_version_(fmt_version&0xF)
  , source_id_(sourceId)
  , bxnum_(bx)
  , lvl1num_(l1num)
{
  bzero(this, sizeInWords()*2);
  source_id_ = sourceId;
  bxnum_ = bx;
  lvl1num_ = l1num;
  format_version_ = fmt_version & 0xF;
  init();
}


void CSCDDUHeader::init() 
{
  bit64_ = 5;
  header2_2_ = 0x0001;
  header2_1_ = header2_3_ = 0x8000;
}


void CSCDDUHeader::setDMBDAV(int dduInput) 
{
  // Set appropriate bit in dmb_dav_

  dmb_dav_ |= ((1 << dduInput) & 0x7FFF);  // dduInput is 0-14

  live_cscs_ |= ((1 << dduInput) & 0x7FFF); // Set DDU Inputs Connected to "Live" CSCs 

  // Count bits set in dmb_dav_... for the trick used see
  // http://en.wikipedia.org/wiki/Hamming_weight or http://graphics.stanford.edu/~seander/bithacks.html

  ncsc_ = 0;
  unsigned short dmbdav = dmb_dav_;
  for( ; dmbdav; ++ncsc_ )
  {
    dmbdav &= dmbdav - 1;
  }
}

bool CSCDDUHeader::check() const 
{
  //std::cout <<"SANDRIK"<<std::hex <<header2_1_<<" "<<header2_2_ <<" "<<header2_3_<<std::endl;
  return bit64_ == 5 && header2_1_ == 0x8000 && header2_3_ == 0x8000
  && header2_2_ == 0x0001;
}

