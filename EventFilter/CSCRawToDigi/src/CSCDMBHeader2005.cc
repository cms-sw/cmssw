#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader2005.h"
#include <iostream>


CSCDMBHeader2005::CSCDMBHeader2005() 
{
  bzero(data(), sizeInWords()*2);
  bits.ddu_code_1 = bits.ddu_code_2 = bits.ddu_code_3 = bits.ddu_code_4 = 0xA;
  bits.newddu_code_1 = bits.newddu_code_2 = bits.newddu_code_3 = bits.newddu_code_4 = 0x9;
}

CSCDMBHeader2005::CSCDMBHeader2005(unsigned short * buf) 
{
  memcpy(data(), buf, sizeInWords()*2);
}


unsigned CSCDMBHeader2005::cfebMovlp() const 
{
  return bits.cfeb_movlp;
}


unsigned CSCDMBHeader2005::dmbCfebSync() const 
{
  return bits.dmb_cfeb_sync;
}

unsigned CSCDMBHeader2005::activeDavMismatch() const 
{
  return bits.active_dav_mismatch;
}


unsigned CSCDMBHeader2005::cfebAvailable() const 
{
  return bits.cfeb_dav;
}


unsigned CSCDMBHeader2005::nalct() const 
{
  return bits.alct_dav_1;
}

unsigned CSCDMBHeader2005::nclct() const 
{
  return bits.tmb_dav_1;
}

unsigned CSCDMBHeader2005::crateID() const 
{
  return bits.dmb_crate;
}

unsigned CSCDMBHeader2005::dmbID() const 
{
  return bits.dmb_id;
}

unsigned CSCDMBHeader2005::bxn() const 
{
  return bits.dmb_bxn;
} 

unsigned CSCDMBHeader2005::bxn12() const
{
  return bits.dmb_bxn1;
}




unsigned CSCDMBHeader2005::l1a() const 
{
  return bits.dmb_l1a;
} 

unsigned CSCDMBHeader2005::l1a24() const
{
  return (bits.dmb_l1a_lowo | (bits.dmb_l1a_hiwo << 12)) ;
}


void CSCDMBHeader2005::setL1A(int l1a) 
{
  bits.dmb_l1a = l1a;
}

void CSCDMBHeader2005::setL1A24(int l1a)
{
  bits.dmb_l1a_lowo = l1a & 0xFFF;
  bits.dmb_l1a_hiwo = (l1a>>12) & 0xFFF;
}


void CSCDMBHeader2005::setBXN(int bxn) 
{
  bits.dmb_bxn = bxn & 0x3F;
  bits.dmb_bxn1 = bxn & 0xFFF;
} 


void CSCDMBHeader2005::setCrateAddress(int crate, int dmbId) 
{
    this->bits.dmb_crate = crate;
    this->bits.dmb_id = dmbId;
}

unsigned CSCDMBHeader2005::sizeInWords() const 
{
  return 8;
}

/// counts from zero
bool CSCDMBHeader2005::cfebAvailable(unsigned icfeb)
{
  assert (icfeb < 5);
  return (cfebAvailable() >> icfeb) & 1;
}

unsigned CSCDMBHeader2005::format_version() const
{ 
  return 0;
}

void CSCDMBHeader2005::addCFEB(int icfeb) 
{
  assert(icfeb < 5);
  bits.cfeb_dav |= (1 << icfeb);
  bits.cfeb_active |= (1 << icfeb);
}

void CSCDMBHeader2005::addNCLCT() 
{
  bits.tmb_dav_1 = bits.tmb_dav_2 = bits.tmb_dav_4 = 1;
}

void CSCDMBHeader2005::addNALCT() 
{
  bits.alct_dav_1 = bits.alct_dav_2 = bits.alct_dav_4 = 1;
}


bool CSCDMBHeader2005::check() const 
{
    return (bits.ddu_code_1==0xA && bits.ddu_code_2==0xA && 
	    bits.ddu_code_3==0xA && bits.ddu_code_4==0xA && 
	    bits.newddu_code_1==0x9 && bits.newddu_code_2==0x9 &&  
	    bits.newddu_code_3==0x9 && bits.newddu_code_4==0x9);
}

