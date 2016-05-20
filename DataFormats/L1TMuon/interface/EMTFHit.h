// Class for input trigger primitives to EMTF - AWB 04.01.16
// Based on L1Trigger/L1TMuon/interface/deprecate/MuonTriggerPrimitive.h
// In particular, see struct CSCData

#ifndef __l1t_EMTFHit_h__
#define __l1t_EMTFHit_h__

#include <vector>
#include <boost/cstdint.hpp> 
#include <cmath>
#include <iostream>

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/L1TMuon/interface/EMTF/ME.h"

namespace l1t {
  
  class EMTFHit {
  public:
    
  EMTFHit() :
    
    // Using -999 instead of -99 b/c this seems most common in the emulator.  Unfortunate. - AWB 17.03.16
    endcap(-999), station(-999), ring(-999), sector(-999), sector_index(-999), subsector(-999), chamber(-999), csc_ID(-999), 
      neighbor(-999), mpc_link(-999), wire(-999), strip(-999), track_num(-999), quality(-999), pattern(-999), bend(-999), 
      valid(-999), sync_err(-999), bc0(-999), bx(-999), stub_num(-999), is_CSC_hit(-999), is_RPC_hit(-999)
      {};
    
    virtual ~EMTFHit() {};

    void ImportCSCDetId (const CSCDetId& _detId);
    CSCDetId CreateCSCDetId();
    void ImportCSCCorrelatedLCTDigi (const CSCCorrelatedLCTDigi& _digi);
    CSCCorrelatedLCTDigi CreateCSCCorrelatedLCTDigi();
    void ImportME (const emtf::ME _ME );

    void PrintSimulatorHeader();
    void PrintForSimulator();

    void SetCSCDetId         (CSCDetId id)                 { csc_DetId         = id;        }
    void SetCSCLCTDigi       (CSCCorrelatedLCTDigi digi)   { csc_LCTDigi       = digi;      }
    
    CSCDetId CSC_DetId                          () const { return csc_DetId;    }
    CSCCorrelatedLCTDigi CSC_LCTDigi            () const { return csc_LCTDigi;  }
    const CSCDetId * PtrCSC_DetId               () const { return &csc_DetId;   }
    const CSCCorrelatedLCTDigi * PtrCSC_LCTDigi () const { return &csc_LCTDigi; }

    void set_endcap         (int  bits) { endcap        = bits; }
    void set_station        (int  bits) { station       = bits; }
    void set_ring           (int  bits) { ring          = bits; }
    void set_sector         (int  bits) { sector        = bits; }
    void set_sector_index   (int  bits) { sector_index  = bits; }
    void set_subsector      (int  bits) { subsector     = bits; }
    void set_chamber        (int  bits) { chamber       = bits; }
    void set_csc_ID         (int  bits) { csc_ID        = bits; }
    void set_neighbor       (int  bits) { neighbor      = bits; }
    void set_mpc_link       (int  bits) { mpc_link      = bits; }
    void set_wire           (int  bits) { wire          = bits; }
    void set_strip          (int  bits) { strip         = bits; }
    void set_track_num      (int  bits) { track_num     = bits; }
    void set_quality        (int  bits) { quality       = bits; }
    void set_pattern        (int  bits) { pattern       = bits; }
    void set_bend           (int  bits) { bend          = bits; }
    void set_valid          (int  bits) { valid         = bits; }
    void set_sync_err       (int  bits) { sync_err      = bits; }
    void set_bc0            (int  bits) { bc0           = bits; }
    void set_bx             (int  bits) { bx            = bits; }
    void set_stub_num       (int  bits) { stub_num      = bits; }
    void set_is_CSC_hit     (int  bits) { is_CSC_hit    = bits; }
    void set_is_RPC_hit     (int  bits) { is_RPC_hit    = bits; }

    int   Endcap         ()  const { return endcap   ;      }
    int   Station        ()  const { return station  ;      }
    int   Ring           ()  const { return ring     ;      }
    int   Sector         ()  const { return sector   ;      }
    int   Sector_index   ()  const { return sector_index;   }
    int   Subsector      ()  const { return subsector;      }
    int   Chamber        ()  const { return chamber  ;      }
    int   CSC_ID         ()  const { return csc_ID   ;      }
    int   Neighbor       ()  const { return neighbor ;      }
    int   MPC_link       ()  const { return mpc_link ;      }
    int   Wire           ()  const { return wire     ;      }
    int   Strip          ()  const { return strip    ;      }
    int   Track_num      ()  const { return track_num;      }
    int   Quality        ()  const { return quality  ;      }
    int   Pattern        ()  const { return pattern  ;      }
    int   Bend           ()  const { return bend     ;      }
    int   Valid          ()  const { return valid    ;      }
    int   Sync_err       ()  const { return sync_err ;      }
    int   BC0            ()  const { return bc0      ;      }
    int   BX             ()  const { return bx       ;      }
    int   Stub_num       ()  const { return stub_num ;      }
    int   Is_CSC_hit     ()  const { return is_CSC_hit;     }
    int   Is_RPC_hit     ()  const { return is_RPC_hit;     }


  private:
    
    CSCDetId csc_DetId;
    CSCCorrelatedLCTDigi csc_LCTDigi;
    
    int   endcap;       // -1 or 1.  Filled in EMTFHit.cc from CSCDetId, modified
    int   station;      //  1 -  4.  Filled in EMTFHit.cc from CSCDetId
    int   ring;         //  1 -  3.  Filled in EMTFHit.cc from CSCDetId
    int   sector;       //  1 -  6.  Filled in EMTFHit.cc from CSCDetId
    int   sector_index; //  0 - 11.  0 - 5 for positive endcap, 6 - 11 for negative.  If a neighbor hit, set by the sector that received it, not the actual sector of the hit.
    int   subsector;    //  1 -  2.  Filled in EMTFHit.cc or emulator using calc_subsector above
    int   chamber;      //  1 - 36.  Filled in EMTFHit.cc from CSCDetId
    int   csc_ID;       //  1 -  9.  Filled in EMTFHit.cc from CSCCorrelatedLCTDigi or emulator from CSCData
    int   neighbor;     //  0 or 1.  Filled in EMTFBlockME.cc 
    int   mpc_link;     //  1 -  3.  Filled in EMTFHit.cc from CSCCorrelatedLCTDigi
    int   wire;         //  1 -  ?.  Filled in EMTFHit.cc from CSCCorrelatedLCTDigi
    int   strip;        //  1 -  ?.  Filled in EMTFHit.cc from CSCCorrelatedLCTDigi
    int   track_num;    //  ? -  ?.  Filled in emulator from CSCData 
    int   quality;      //  0 - 15.  Filled in EMTFHit.cc from CSCCorrelatedLCTDigi
    int   pattern;      //  0 - 10.  Filled in EMTFHit.cc from CSCCorrelatedLCTDigi
    int   bend;         //  0 or 1.  Filled in EMTFHit.cc from CSCCorrelatedLCTDigi
    int   valid;        //  0 or 1.  Filled in EMTFHit.cc from CSCCorrelatedLCTDigi
    int   sync_err;     //  0 or 1.  Filled in EMTFHit.cc from CSCCorrelatedLCTDigi
    int   bc0;          //  0 or 1.
    int   bx;           //  -3 - 3.  Filled in EMTFHit.cc from CSCCorrelatedLCTDigi
    int   stub_num;     //  0 or 1.
    int   is_CSC_hit;   //  0 or 1.  Filled in EMTFHit.cc
    int   is_RPC_hit;   //  0 or 1.  Filled in EMTFHit.cc

  }; // End of class EMTFHit
  
  // Define a vector of EMTFHit
  typedef std::vector<EMTFHit> EMTFHitCollection;
  
} // End of namespace l1t

#endif /* define __l1t_EMTFHit_h__ */
