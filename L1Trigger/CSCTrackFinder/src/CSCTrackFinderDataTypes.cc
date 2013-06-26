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

pt_address& pt_address::operator=(const unsigned& u)
{
  this->delta_phi_12   = ((1<<8)-1)&u;
  this->delta_phi_23   = ((1<<4)-1)&(u>>8);
  this->track_eta      = ((1<<4)-1)&(u>>12);
  this->track_mode     = ((1<<4)-1)&(u>>16);
  this->delta_phi_sign = ((1<<1)-1)&(u>>20);
  this->track_fr       = ((1<<1)-1)&(u>>21);
  
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

pt_data& pt_data::operator=(const unsigned short& us)
{
  this->front_rank         = ((1<<7)-1)&us;
  this->charge_valid_front = ((1<<1)-1)&(us>>7);
  this->rear_rank          = ((1<<7)-1)&(us>>8);
  this->charge_valid_rear  = ((1<<1)-1)&(us>>15);
  
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

unsigned short pt_data::toint() const
{
  unsigned short us = 0;
  us = front_rank | (charge_valid_front << 7) | (rear_rank << 8) | (charge_valid_rear << 15);
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

unsigned pt_address::toint() const
{
  unsigned u = 0;
  u = delta_phi_12 | (delta_phi_23 << 8) | (track_eta << 12) | (track_mode << 16) | (delta_phi_sign << 20) | (track_fr << 21);
  return u;
}

unsigned pt_address::delta_phi() const
{
  return ( delta_phi_12 | (delta_phi_23 << 8) );
}
