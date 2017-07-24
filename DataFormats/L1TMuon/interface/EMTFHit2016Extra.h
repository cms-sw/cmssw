// Class for input trigger primitives to EMTF - AWB 04.01.16
// Based on L1Trigger/L1TMuon/interface/deprecate/MuonTriggerPrimitive.h
// In particular, see struct CSCData

#ifndef DataFormats_L1TMuon_EMTFHit2016Extra_h
#define DataFormats_L1TMuon_EMTFHit2016Extra_h

#include <vector>
#include <boost/cstdint.hpp> 
#include <cmath>
#include <iostream>

#include "DataFormats/GeometryVector/interface/Pi.h"
#include "DataFormats/L1TMuon/interface/EMTFHit2016.h"

namespace l1t {
  class EMTFHit2016Extra: public EMTFHit2016 {
  public:
    
  EMTFHit2016Extra() :
    
      bx0(-999), layer(-999), zone(-999), phi_hit(-999), phi_zone(-999), phi_loc_int(-999), 
      phi_loc_deg(-999), phi_loc_rad(-999), phi_glob_deg(-999), phi_glob_rad(-999), phi_geom_rad(-999), 
      theta_int(-999), theta_loc(-999), theta_deg(-999), theta_rad(-999), eta(-999)
      {};
    
    virtual ~EMTFHit2016Extra() {};

    void ImportCSCCorrelatedLCTDigi (const CSCCorrelatedLCTDigi& _digi);
    EMTFHit2016 CreateEMTFHit2016();
    EMTFHit2016Extra Clone() {
      EMTFHit2016Extra ht;
      ht.set_endcap(Endcap()); ht.set_station(Station()); ht.set_ring(Ring()); ht.set_sector(Sector()); 
      ht.set_sector_index(Sector_index()); ht.set_subsector(Subsector()); ht.set_chamber(Chamber()); 
      ht.set_csc_ID(CSC_ID()); ht.set_roll(Roll()); ht.set_rpc_layer(RPC_layer()); ht.set_neighbor(Neighbor()); 
      ht.set_mpc_link(MPC_link()); ht.set_wire(Wire()); ht.set_strip(Strip()); ht.set_strip_hi(Strip_hi()); 
      ht.set_strip_low(Strip_low()); ht.set_track_num(Track_num()); ht.set_quality(Quality()); ht.set_pattern(Pattern()); 
      ht.set_bend(Bend()); ht.set_valid(Valid()); ht.set_sync_err(Sync_err()); ht.set_bc0(BC0()); ht.set_bx(BX()); 
      ht.set_stub_num(Stub_num()); ht.set_is_CSC_hit(Is_CSC_hit()); ht.set_is_RPC_hit(Is_RPC_hit()); 

      ht.SetCSCDetId(CSC_DetId()); ht.SetRPCDetId(RPC_DetId()); ht.SetCSCLCTDigi(CSC_LCTDigi()); ht.SetRPCDigi(RPC_Digi());

      ht.set_bx0(BX0()); ht.set_layer(Layer()); ht.set_zone(Zone()); ht.set_phi_hit(Phi_hit()); ht.set_phi_zone(Phi_zone()); 
      ht.set_phi_loc_int(Phi_loc_int()); ht.set_phi_loc_deg(Phi_loc_deg()); ht.set_phi_loc_rad(Phi_loc_rad()); 
      ht.set_phi_glob_deg(Phi_glob_deg()); ht.set_phi_glob_rad(Phi_glob_rad()); ht.set_phi_geom_rad(Phi_geom_rad()); 
      ht.set_theta_int(Theta_int()); ht.set_theta_loc(Theta_loc()); ht.set_theta_deg(Theta_deg()); ht.set_theta_rad(Theta_rad()); 
      ht.set_eta(Eta()); 
      return ht;
    }
    void SetZoneContribution (std::vector<int> vect_ints)  { zone_contribution = vect_ints; }
    std::vector<int> Zone_contribution          () const { return zone_contribution; }

    void set_bx0            (int  bits) { bx0           = bits; }
    void set_layer          (int  bits) { layer         = bits; }
    void set_zone           (int  bits) { zone          = bits; }
    void set_phi_hit        (int  bits) { phi_hit       = bits; }
    void set_phi_zone       (int  bits) { phi_zone      = bits; }
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
    int   Zone           ()  const { return zone;           }
    int   Phi_hit        ()  const { return phi_hit;        }
    int   Phi_zone       ()  const { return phi_zone;       }
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
    
    int   bx0;          //  1-3600.  Filled in EMTFHit2016.cc from CSCCorrelatedLCTDigi
    int   layer;
    int   zone;         //  4 - 118. Filled in emulator from ConvertedHit.Zhit()
    int   phi_hit;      //  1 - 42.  Filled in emulator from ConvertedHit.Ph_hit()
    int   phi_zone;     //  1 -  6.  Filled in emulator from ConvertedHit.Phzvl()
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

  }; // End of class EMTFHit2016Extra
  
  // Define a vector of EMTFHit2016Extra
  typedef std::vector<EMTFHit2016Extra> EMTFHit2016ExtraCollection;
  
} // End of namespace l1t

#endif /* define DataFormats_L1TMuon_EMTFHit2016Extra_h */
