#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader2013.h"
#include <iostream>


CSCDMBHeader2013::CSCDMBHeader2013() 
{
  bzero(data(), sizeInWords()*2);
  bits.ddu_code_1 = bits.ddu_code_2 = bits.ddu_code_3 = bits.ddu_code_4 = 0xA;
  bits.newddu_code_1 = bits.newddu_code_2 = bits.newddu_code_3 = bits.newddu_code_4 = 0x9;
}

CSCDMBHeader2013::CSCDMBHeader2013(unsigned short * buf) 
{
  memcpy(data(), buf, sizeInWords()*2);
}


unsigned CSCDMBHeader2013::cfebMovlp() const 
{
  return bits.cfeb_movlp;
}


unsigned CSCDMBHeader2013::dmbCfebSync() const 
{
  return bits.dmb_cfeb_sync;
}

unsigned CSCDMBHeader2013::activeDavMismatch() const 
{
  return bits.clct_dav_mismatch;
}

unsigned CSCDMBHeader2013::format_version() const
{
  return bits.fmt_version;
}


unsigned CSCDMBHeader2013::cfebAvailable() const 
{
  return bits.cfeb_dav;
}


unsigned CSCDMBHeader2013::nalct() const 
{
  return bits.alct_dav;
}

unsigned CSCDMBHeader2013::nclct() const 
{
  return bits.tmb_dav;
}

unsigned CSCDMBHeader2013::crateID() const 
{
  return bits.dmb_crate;
}

unsigned CSCDMBHeader2013::dmbID() const 
{
  return bits.dmb_id;
}

unsigned CSCDMBHeader2013::bxn() const 
{
  return bits.dmb_bxn;
} 

unsigned CSCDMBHeader2013::bxn12() const
{
  return bits.dmb_bxn1;
}

unsigned CSCDMBHeader2013::l1a() const 
{
  return bits.dmb_l1a;
} 

unsigned CSCDMBHeader2013::l1a24() const
{
  return (bits.dmb_l1a_lowo | (bits.dmb_l1a_hiwo << 12)) ;
}

void CSCDMBHeader2013::setL1A(int l1a) 
{
  bits.dmb_l1a = l1a & 0x1F;
}

void CSCDMBHeader2013::setL1A24(int l1a)
{
  bits.dmb_l1a_lowo = l1a & 0xFFF;
  bits.dmb_l1a_hiwo = (l1a>>12) & 0xFFF;
}


void CSCDMBHeader2013::setBXN(int bxn) 
{
  bits.dmb_bxn1 = bxn & 0xFFF;
  bits.dmb_bxn = bxn & 0x1F;
} 


void CSCDMBHeader2013::setCrateAddress(int crate, int dmbId) 
{
    this->bits.dmb_crate = crate;
    this->bits.dmb_id = dmbId;
}

unsigned CSCDMBHeader2013::sizeInWords() const 
{
  return 8;
}

/// counts from zero
bool CSCDMBHeader2013::cfebAvailable(unsigned icfeb)
{
  assert (icfeb < 7);
  return (cfebAvailable() >> icfeb) & 1;
}

void CSCDMBHeader2013::addCFEB(int icfeb) 
{
  assert(icfeb < 7);
  bits.cfeb_dav |= (1 << icfeb);
  bits.cfeb_clct_sent |= (1 << icfeb);
}

void CSCDMBHeader2013::addNCLCT() 
{
  bits.tmb_dav =  bits.tmb_dav_copy =  bits.tmb_dav_copy2 = 1;
}

void CSCDMBHeader2013::addNALCT() 
{
  bits.alct_dav = bits.alct_dav_copy = bits.alct_dav_copy2 = 1;
}


bool CSCDMBHeader2013::check() const 
{
    return (bits.ddu_code_1==0xA && bits.ddu_code_2==0xA && 
	    bits.ddu_code_3==0xA && bits.ddu_code_4==0xA && 
	    bits.newddu_code_1==0x9 && bits.newddu_code_2==0x9 &&  
	    bits.newddu_code_3==0x9 && bits.newddu_code_4==0x9);
}

