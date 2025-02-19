#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include <iostream>


CSCDMBHeader::CSCDMBHeader() 
{
  bzero(this, sizeInWords()*2);
  ddu_code_1 = ddu_code_2 = ddu_code_3 = ddu_code_4 = 0xA;
  newddu_code_1 = newddu_code_2 = newddu_code_3 = newddu_code_4 = 0x9;
}

CSCDMBHeader::CSCDMBHeader(unsigned short * buf) 
{
  memcpy(this, buf, sizeInWords()*2);
}


unsigned CSCDMBHeader::cfebMovlp() const 
{
  return cfeb_movlp;
}


unsigned CSCDMBHeader::dmbCfebSync() const 
{
  return dmb_cfeb_sync;
}

unsigned CSCDMBHeader::activeDavMismatch() const 
{
  return active_dav_mismatch;
}


unsigned CSCDMBHeader::cfebAvailable() const 
{
  return cfeb_dav;
}


unsigned CSCDMBHeader::nalct() const 
{
  return alct_dav_1;
}

unsigned CSCDMBHeader::nclct() const 
{
  return tmb_dav_1;
}

unsigned CSCDMBHeader::crateID() const 
{
  return dmb_crate;
}

unsigned CSCDMBHeader::dmbID() const 
{
  return dmb_id;
}

unsigned CSCDMBHeader::bxn() const 
{
  return dmb_bxn;
} 

unsigned CSCDMBHeader::bxn12() const
{
  return dmb_bxn1;
}




unsigned CSCDMBHeader::l1a() const 
{
  return dmb_l1a;
} 


void CSCDMBHeader::setL1A(int l1a) 
{
  dmb_l1a = l1a;
}

void CSCDMBHeader::setBXN(int bxn) 
{
  dmb_bxn = bxn;
} 


void CSCDMBHeader::setCrateAddress(int crate, int dmbId) 
{
    this->dmb_crate = crate;
    this->dmb_id = dmbId;
}

unsigned CSCDMBHeader::sizeInWords() const 
{
  return 8;
}

/// counts from zero
bool CSCDMBHeader::cfebAvailable(unsigned icfeb)
{
  assert (icfeb < 5);
  return (cfebAvailable() >> icfeb) & 1;
}

void CSCDMBHeader::addCFEB(int icfeb) 
{
  assert(icfeb < 5);
  cfeb_dav |= (1 << icfeb);
}

void CSCDMBHeader::addNCLCT() 
{
  tmb_dav_1 =  tmb_dav_2 =  tmb_dav_4 = 1;
}

void CSCDMBHeader::addNALCT() 
{
  alct_dav_1 = alct_dav_2 = alct_dav_4 = 1;
}


bool CSCDMBHeader::check() const 
{
    return (ddu_code_1==0xA && ddu_code_2==0xA && 
	    ddu_code_3==0xA && ddu_code_4==0xA && 
	    newddu_code_1==0x9 && newddu_code_2==0x9 &&  
	    newddu_code_3==0x9 && newddu_code_4==0x9);
}

