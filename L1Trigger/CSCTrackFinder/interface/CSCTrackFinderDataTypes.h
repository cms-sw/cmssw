#ifndef CSCTrackFinder_CSCTrackFinderDataTypes_h
#define CSCTrackFinder_CSCTrackFinderDataTypes_h

// Address Types                                                                                      
typedef class local_phi_address
{
 public:
  unsigned int strip        : 8;
  unsigned int clct_pattern : 3;
  unsigned int pattern_type : 1; // 1 is half strip 0 is di strip
  unsigned int quality      : 4;
  unsigned int lr           : 1;
  unsigned int spare        : 2;
  unsigned int zero         : 13;

  local_phi_address(): strip(0), 
                       clct_pattern(0), 
                       pattern_type(0), 
                       quality(0), 
                       lr(0), 
                       spare(0), 
                       zero(0) {};
  local_phi_address(const unsigned& u) { this->operator=(u); }
    
  local_phi_address& operator=(const unsigned& u);
  unsigned toint() const;
    
} lclphiadd;

typedef class global_phi_address
{
 public:
  unsigned int phi_local    : 10;
  unsigned int wire_group   : 5;  // bits 2-6 of wg                                                   
  unsigned int cscid        : 4;
  unsigned int zero         : 13;
  
  global_phi_address(): phi_local(0), 
                        wire_group(0), 
                        cscid(0), 
                        zero(0) {};
  global_phi_address(const unsigned& u) { this->operator=(u); }

  global_phi_address& operator=(const unsigned& u);
  unsigned toint() const;

} gblphiadd;

typedef class global_eta_address
{
 public:
  unsigned int phi_bend     : 6;
  unsigned int phi_local    : 2;
  unsigned int wire_group   : 7;
  unsigned int cscid        : 4;
  unsigned int zero         : 13;

  global_eta_address(): phi_bend(0), 
                        phi_local(0), 
                        wire_group(0), 
                        cscid(0), 
                        zero(0) {};
  global_eta_address(const unsigned& u) { this->operator=(u); }

  global_eta_address& operator=(const unsigned& u);
  unsigned toint() const;

} gbletaadd;

typedef class pt_address
{
 public:
  unsigned int delta_phi_12   : 8;
  unsigned int delta_phi_23   : 4;
  unsigned int track_eta      : 4;
  unsigned int track_mode     : 4;
  unsigned int delta_phi_sign : 1;
  unsigned int track_fr       : 1;

  pt_address(): delta_phi_12(0),
                delta_phi_23(0), 
                track_eta(0), 
                track_mode(0), 
                delta_phi_sign(0), 
                track_fr(0)   {};
  pt_address(const unsigned& us) { this->operator=(us); }

  pt_address& operator=(const unsigned&);
  unsigned toint() const;
  
  unsigned delta_phi() const; // for 2 stn track
} ptadd;

/// Data Types                                                                                        
typedef class local_phi_data
{
 public:
  unsigned short phi_local      : 10;
  unsigned short phi_bend_local : 6;

  local_phi_data(): phi_local(0),
                    phi_bend_local(0) {};
  local_phi_data(const unsigned short& us) { this->operator=(us); }

  local_phi_data& operator=(const unsigned short& us);
  unsigned short toint() const;

} lclphidat;

typedef class global_phi_data
{
 public:
  unsigned short global_phi : 12;
  unsigned short spare      : 4;
  
  global_phi_data(): global_phi(0),
                     spare(0) {};  
  global_phi_data(const unsigned short& us) { this->operator=(us); }

  global_phi_data& operator=(const unsigned short& us);
  unsigned short toint() const;

} gblphidat;

typedef class global_eta_data
{
 public:
  unsigned short global_eta  : 7;
  unsigned short global_bend : 5;
  unsigned short spare       : 4;
  
  global_eta_data(): global_eta(0),
                     global_bend(0),
                     spare(0) {};
  global_eta_data(const unsigned short& us) { this->operator=(us); }

  global_eta_data& operator=(const unsigned short& us);
  unsigned short toint() const;

} gbletadat;


typedef class pt_data
{
 public:
  unsigned short front_rank         : 7;
  unsigned short charge_valid_front : 1;
  unsigned short rear_rank          : 7;
  unsigned short charge_valid_rear  : 1;
  
  pt_data(): front_rank(0),
             charge_valid_front(0),
             rear_rank(0),
             charge_valid_rear(0) {};
  pt_data(const unsigned short& us) { this->operator=(us); }

  pt_data& operator=(const unsigned short&);
  unsigned short toint() const;

} ptdat;

#endif
