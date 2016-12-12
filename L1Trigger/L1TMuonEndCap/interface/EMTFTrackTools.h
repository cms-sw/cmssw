
#ifndef EMTFTrackTools_h
#define EMTFTrackTools_h

#include "DataFormats/L1TMuon/interface/EMTFTrackExtra.h"

namespace l1t {

  int calc_uGMT_chamber( int _csc_ID, int _subsector, int _neighbor, int _station);

  // phi_loc gives the exact phi value (marked "phi_full" in the EMTF DAQ format document)  
  // phi_GMT (the value used by GMT) is a rough estimate, with offsets of 1-2 degrees for some phi values
  // The conversion used is: phi_GMT =        (360/576)*phi_GMT_int +        (180/576)
  // More accurate would be: phi_GMT = 1.0208*(360/576)*phi_GMT_int + 1.0208*(180/576) + 0.552
  inline float calc_pt(int bits)                    { return (bits - 1) * 0.5;                                  }
  inline int   calc_pt_GMT(float val)               { return (val * 2) + 1;                                     }
  inline float calc_eta(int bits)                   { return bits * 0.010875;                                   }
  inline int   calc_eta_GMT(float val)              { return val / 0.010875;                                    }
  inline float calc_theta_deg(float _eta)           { return 2*atan( exp(-1*_eta) ) * (180/Geom::pi());         }
  inline float calc_theta_rad(float _eta)           { return 2*atan( exp(-1*_eta) );                            }
  inline float calc_phi_loc_deg(int bits)           { return (bits / 60.0) - 22.0;                              }
  inline float calc_phi_loc_rad(int bits)           { return (bits * Geom::pi() / 10800) - (22 * Geom::pi() / 180); }
  inline int   calc_phi_loc_int(float val)          { return (val + 2) * 60;                                    }
  inline float calc_phi_GMT_deg(int bits)           { return (bits * 0.625) + 0.3125;                           } /* x (360/576) + (180/576) */
  inline float calc_phi_GMT_deg_corr(int bits)      { return (bits * 0.625 * 1.0208) + 0.3125 * 1.0208 + 0.552; } /* AWB mod 09.02.16 */
  inline float calc_phi_GMT_rad(int bits)           { return (bits * Geom::pi() / 288) + (Geom::pi() / 576);    } /* x (2*pi/576) + (pi/576) */
  inline int   calc_phi_GMT_int(float val)          { return (val - 0.3125) / 0.625;                            } /* - (180/576) / (360/576) */
  inline float calc_phi_glob_deg(float loc, int sect) { float tmp = loc + 15 + (sect - 1)*60; return (tmp < 180) ? tmp : tmp - 360; }
  inline float calc_phi_glob_rad(float loc, int sect) { float tmp = loc + (Geom::pi()/12) + (sect - 1)*(Geom::pi()/3); return (tmp < Geom::pi()) ? tmp : tmp - 2*Geom::pi(); }
  inline int   calc_sector_GMT (int _sector)        { return _sector - 1; }
  inline int   calc_sector_from_index(int index)    { return (index < 6) ? index + 1 : index - 5;               }

}

#endif /* define EMTFTrackTools_h */
