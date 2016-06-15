
#ifndef EMTFHitTools_h
#define EMTFHitTools_h

#include "DataFormats/L1TMuon/interface/EMTFHitExtra.h"

namespace l1t {

  int calc_ring (int _station, int _csc_ID, int _strip);
  int calc_chamber (int _station, int _sector, int _subsector, int _ring, int _csc_ID);
  
  inline int   calc_subsector   (int _station, int _chamber ) {  // Why does emulator have the following csc_ID dependence? - AWB 11.04.16
    if ( _station == 1 ) return ( (_chamber % 6) > 2 ) ? 1 : 2;  // "if(station == 1 && id > 3 && id < 7)" - AWB 11.04.16
    else if ( _station < 0 || _chamber < 0 ) return -999;        // Function works because first 3 chambers in sector 1 are 3/4/5,
    else return -1; }                                            // while the last 3 in sector 6 are 36/1/2
  inline float calc_theta_deg_from_int (int _theta_int)   { return _theta_int * 0.2851562 + 8.5; }
  inline float calc_theta_rad_from_int (int _theta_int)   { return (_theta_int * 0.2851562 + 8.5) * (Geom::pi() / 180); }
  inline float calc_eta_from_theta_rad (float _theta_rad) { return -1 * log( tan( _theta_rad / 2 ) ); }
  inline float calc_phi_glob_deg_hit(float loc, int sect_ind) { 
    float tmp = loc + 15 + (sect_ind < 6 ? sect_ind : sect_ind - 6)*60; 
    return (tmp < 180) ? tmp : tmp - 360; }
  inline float calc_phi_glob_rad_hit(float loc, int sect_ind) { 
    float tmp = loc + (Geom::pi()/12) + (sect_ind < 6 ? sect_ind : sect_ind - 6)*(Geom::pi()/3); 
    return (tmp < Geom::pi()) ? tmp : tmp - 2*Geom::pi(); }

} // End namespace l1t

#endif /* define EMTFHitTools_h */
