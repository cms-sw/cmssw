#include "EventFilter/CSCRawToDigi/interface/CSCDDUHeader.h"
#include <string.h> // for bzero
#include <iostream>

CSCDDUHeader::CSCDDUHeader() 
{
  bzero(this, sizeInWords()*2);
  init();
}


CSCDDUHeader::CSCDDUHeader(unsigned bx, unsigned l1num, unsigned sourceId) 
  : source_id_(sourceId), bxnum_(bx), lvl1num_(l1num)
{
  bzero(this, sizeInWords()*2);
  source_id_ = sourceId;
  bxnum_ = bx;
  lvl1num_ = l1num;
  init();
}


void CSCDDUHeader::init() 
{
  bit64_ = 5;
  header2_2_ = 0x0001;
  header2_1_ = header2_3_ = 0x8000;
}


void CSCDDUHeader::setDMBDAV(int dmbId) 
{
  dmb_dav_ |= (1 << dmbId); 
  // use Ritchie algo to count bits
  ncsc_ = 0;
  unsigned short dmbdav = dmb_dav_;
  for( ; dmbdav; ++ncsc_)
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

