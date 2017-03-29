// Class for input trigger primitives to EMTF - AWB 04.01.16
// Based on L1Trigger/L1TMuon/interface/deprecate/MuonTriggerPrimitive.h
// In particular, see struct CSCData

#ifndef __l1t_EMTFHitExtra_h__
#define __l1t_EMTFHitExtra_h__

#include <vector>
#include <boost/cstdint.hpp> 
#include <cmath>
#include <iostream>

#include "DataFormats/GeometryVector/interface/Pi.h"
#include "DataFormats/L1TMuon/interface/EMTFHit.h"

namespace l1t {
  class EMTFHitExtra: public EMTFHit {
  public:
    
  EMTFHitExtra() :
    
      bx0(-999), layer(-999), zone_hit(-999), phi_hit(-999), phi_z_val(-999), phi_loc_int(-999), 
      phi_loc_deg(-999), phi_loc_rad(-999), phi_glob_deg(-999), phi_glob_rad(-999), phi_geom_rad(-999), 
      theta_int(-999), theta_loc(-999), theta_deg(-999), theta_rad(-999), eta(-999)
      {};
    
    virtual ~EMTFHitExtra() {};

    void ImportCSCCorrelatedLCTDigi (const CSCCorrelatedLCTDigi& _digi);
    EMTFHit CreateEMTFHit();

    void SetZoneContribution (std::vector<int> vect_ints)  { zone_contribution = vect_ints; }
    std::vector<int> Zone_contribution          () const { return zone_contribution; }

    void set_bx0            (int  bits) { bx0           = bits; }
    void set_layer          (int  bits) { layer         = bits; }
    void set_zone_hit       (int  bits) { zone_hit      = bits; }
    void set_phi_hit        (int  bits) { phi_hit       = bits; }
    void set_phi_z_val      (int  bits) { phi_z_val     = bits; }
    void set_phi_loc_int    (int  bits) { phi_loc_int   = bits; }
    void set_phi_loc_deg    (float val) { phi_loc_deg   = val;  }
    void set_phi_loc_rad    (float val) { phi_loc_rad   = val;  }
    void set_phi_glob_deg   (float val) { (val < 180) ? phi_glob_deg = val : phi_glob_deg = val - 360;  }
    void set_phi_glob_rad   (float val) { (val < Geom::pi() ) ? phi_glob_rad = val : phi_glob_rad = val - 2*Geom::pi(); }
    void set_phi_geom_rad   (float val) { phi_geom_rad   = val; }
    void set_theta_int      (int  bits) { theta_int     = bits; }
    void set_theta_loc      (float val) { theta_loc      = val; }
    void set_theta_deg      (float val) { theta_deg      = val; }
    void set_theta_rad      (float val) { theta_rad      = val; }
    void set_eta            (float val) { eta            = val; }

    int   BX0            ()  const { return bx0;            }
    int   Layer          ()  const { return layer;          }
    int   Zone_hit       ()  const { return zone_hit;       }
    int   Phi_hit        ()  const { return phi_hit;        }
    int   Phi_Z_val      ()  const { return phi_z_val;      }
    int   Phi_loc_int    ()  const { return phi_loc_int;    }
    float Phi_loc_deg    ()  const { return phi_loc_deg;    }
    float Phi_loc_rad    ()  const { return phi_loc_rad;    }
    float Phi_glob_deg   ()  const { return phi_glob_deg;   }
    float Phi_glob_rad   ()  const { return phi_glob_rad;   }
    float Phi_geom_rad   ()  const { return phi_geom_rad;   }
    int   Theta_int      ()  const { return theta_int;      }
    float Theta_loc      ()  const { return theta_loc;      }
    float Theta_deg      ()  const { return theta_deg;      }
    float Theta_rad      ()  const { return theta_rad;      }
    float Eta            ()  const { return eta;            }


  private:
    
    std::vector<int> zone_contribution; // Filled in emulator from ConvertedHit.ZoneContribution()
    
    int   bx0;          //  1-3600.  Filled in EMTFHit.cc from CSCCorrelatedLCTDigi
    int   layer;
    int   zone_hit;     //  4 - 118. Filled in emulator from ConvertedHit.Zhit()
    int   phi_hit;      //  1 - 42.  Filled in emulator from ConvertedHit.Ph_hit()
    int   phi_z_val;    //  1 -  6.  Filled in emulator from ConvertedHit.Phzvl()
    int   phi_loc_int;  //  ? -  ?.  Filled in emulator from ConvertedHit.Phi()
    float phi_loc_deg;  //  ? -  ?.  Filled in emulator, calculated from phi_loc_int with GetPackedPhi
    float phi_loc_rad;  //  ? -  ?.  Filled in emulator, calculated from phi_loc_int with GetPackedPhi
    float phi_glob_deg; //  +/-180.  Filled in emulator, calculated from phi_loc_int with GetPackedPhi
    float phi_glob_rad; //  +/- pi.  Filled in emulator, calculated from phi_loc_int with GetPackedPhi
    float phi_geom_rad; //  The global phi value returned by L1Trigger/L1TMuon/interface/deprecate/GeometryTranslator.h.  Not yet filled - AWB 06.04.16
    int   theta_int;    //  ? -  ?.  Filled in emulator from ConvertedHit.Theta()
    float theta_loc;    //  Some bizzare local definition of theta.  Not yet filled - AWB 06.04.16
    float theta_deg;    // 10 - 45.  Filled in emulator from calc_theta_deg above
    float theta_rad;    // .2 - .8.  Filled in emulator from calc_theta_rad above
    float eta;          // +/- 2.5.  Filled in emulator from calc_eta above

  }; // End of class EMTFHitExtra
  
  // Define a vector of EMTFHitExtra
  typedef std::vector<EMTFHitExtra> EMTFHitExtraCollection;
  
} // End of namespace l1t

#endif /* define __l1t_EMTFHitExtra_h__ */
