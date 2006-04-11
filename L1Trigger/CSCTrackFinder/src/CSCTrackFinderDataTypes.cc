#include <L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h>

local_phi_address& local_phi_address::operator=(const unsigned& u)
{
  this->strip = ((1<<8)-1)&u;
  this->clct_pattern = ((1<<3)-1)&(u>>8);
  this->pattern_type = 1&(u>>11);
  this->quality = ((1<<4)-1)&(u>>12);
  this->lr = 1&(u>>16);
  return *this;
}

global_phi_address& global_phi_address::operator=(const unsigned& u)
{
  this->phi_local = ((1<<10)-1)&u;
  this->wire_group = ((1<<5)-1)&(u>>10);
  this->cscid = ((1<<4)-1)&(u>>15);
  return *this;
}

global_eta_address& global_eta_address::operator=(const unsigned& u)
{
  this->phi_bend = ((1<<6)-1)&u;
  this->phi_local = ((1<<2)-1)&(u>>6);
  this->wire_group = ((1<<7)-1)&(u>>8);
  this->cscid = ((1<<4)-1)&(u>>15);
  return *this;
}

local_phi_data& local_phi_data::operator=(const unsigned short& us)
{
  this->phi_local = ((1<<10)-1)&us;
  this->phi_bend_local = ((1<<6)-1)&(us>>10);
  return *this;
}

global_phi_data& global_phi_data::operator=(const unsigned short& us)
{   
  this->global_phi = ((1<<12)-1)&us;
  return *this;
}

global_eta_data& global_eta_data::operator=(const unsigned short& us)
{    
  this->global_eta = ((1<<7)-1)&us;
  this->global_bend = ((1<<5)-1)&(us>>7);
  return *this;
}

unsigned short local_phi_data::toint() const
{
  unsigned short us = 0;
  us = (phi_local | (phi_bend_local << 10));
  return us;
}

unsigned short global_eta_data::toint() const
{
  unsigned short us = 0;
  us = (global_eta | (global_bend << 7));
  return us;
}

unsigned short global_phi_data::toint() const
{  
  unsigned short us = 0;
  us = global_phi;
  return us;
}

unsigned local_phi_address::toint() const
{
  unsigned u = 0;
  u = strip | (clct_pattern << 8) | (pattern_type << 11) | (quality << 12) | (lr << 16);
  return u;
}

unsigned global_eta_address::toint() const
{
  unsigned u = 0;
  u = phi_bend | (phi_local << 6) | (wire_group << 8) | (cscid << 15);
  return u;
}

unsigned global_phi_address::toint() const
{
  unsigned u = 0;
  u = phi_local | (wire_group << 10) | (cscid << 15);
  return u;
}
