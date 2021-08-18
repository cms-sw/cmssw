// Class for input trigger primitives to EMTF - AWB 04.01.16
// Based on L1Trigger/L1TMuon/interface/MuonTriggerPrimitive.h
// In particular, see struct CSCData

#ifndef DataFormats_L1TMuon_EMTFHit_h
#define DataFormats_L1TMuon_EMTFHit_h

#include <cstdint>
#include <vector>

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCluster.h"
#include "DataFormats/L1TMuon/interface/CPPFDigi.h"
#include "DataFormats/L1TMuon/interface/L1TMuonSubsystems.h"

namespace l1t {

  namespace l1tmu = L1TMuon;

  class EMTFHit {
  public:
    EMTFHit()
        : rawDetId(0),
          subsystem(-99),
          endcap(-99),
          station(-99),
          ring(-99),
          sector(-99),
          sector_RPC(-99),
          sector_idx(-99),
          subsector(-99),
          subsector_RPC(-99),
          chamber(-99),
          csc_ID(-99),
          csc_nID(-99),
          roll(-99),
          neighbor(-99),
          mpc_link(-99),
          pc_sector(-99),
          pc_station(-99),
          pc_chamber(-99),
          pc_segment(-99),
          wire(-99),
          strip(-99),
          strip_hi(-99),
          strip_low(-99),
          strip_quart(-99),       // Run 3
          strip_eighth(-99),      // Run 3
          strip_quart_bit(-99),   // Run 3
          strip_eighth_bit(-99),  // Run 3
          track_num(-99),
          quality(-99),
          pattern(-99),
          pattern_run3(-99),  // Run 3
          bend(-99),
          slope(-99),  // Run 3
          valid(-99),
          sync_err(-99),
          layer(-99),  // TODO: verify inclusion for GEM, or better to generalize this class... - JS 06.07.20
          bc0(-99),
          bx(-99),
          stub_num(-99),
          phi_fp(-99),
          theta_fp(-99),
          zone_hit(-99),
          zone_code(-99),
          fs_segment(-99),
          fs_zone_code(-99),
          bt_station(-99),
          bt_segment(-99),
          phi_loc(-99),
          phi_glob(-999),
          theta(-99),
          eta(-99),
          time(-99),
          phi_sim(-999),
          theta_sim(-99),
          eta_sim(-99),
          rho_sim(-99),
          z_sim(-99),
          alct_quality(-99),
          clct_quality(-99){};

    virtual ~EMTFHit(){};

    CSCDetId CreateCSCDetId() const;
    RPCDetId CreateRPCDetId() const;
    GEMDetId CreateGEMDetId() const;
    ME0DetId CreateME0DetId() const;

    // void ImportCSCCorrelatedLCTDigi (const CSCCorrelatedLCTDigi& _digi);
    CSCCorrelatedLCTDigi CreateCSCCorrelatedLCTDigi() const;
    // void ImportRPCDigi (const RPCDigi& _digi);
    // RPCDigi CreateRPCDigi() const;
    // void ImportCPPFDigi (const CPPFDigi& _digi);
    CPPFDigi CreateCPPFDigi() const;
    // void ImportGEMPadDigiCluster (const GEMPadDigiCluster& _digi);  // TODO: implement placeholder when others are implemented
    GEMPadDigiCluster CreateGEMPadDigiCluster() const;

    // void PrintSimulatorHeader() const;
    // void PrintForSimulator() const;

    //void SetCSCDetId   (const CSCDetId& id)                 { csc_DetId         = id;        }
    //void SetRPCDetId   (const RPCDetId& id)                 { rpc_DetId         = id;        }
    //void SetGEMDetId   (const GEMDetId& id)                 { gem_DetId         = id;        }
    //void SetCSCLCTDigi (const CSCCorrelatedLCTDigi& digi)   { csc_LCTDigi       = digi;      }
    //void SetRPCDigi    (const RPCDigi& digi)                { rpc_Digi          = digi;      }
    //void SetCPPFDigi   (const CPPFDigi& digi)               { cppf_Digi         = digi;      }
    //void SetGEMPadDigiCluster (const GEMPadDigiCluster& digi)             { gem_PadClusterDigi       = digi;      }
    void SetCSCDetId(const CSCDetId& id) { rawDetId = id.rawId(); }
    void SetRPCDetId(const RPCDetId& id) { rawDetId = id.rawId(); }
    void SetGEMDetId(const GEMDetId& id) { rawDetId = id.rawId(); }
    void SetME0DetId(const ME0DetId& id) { rawDetId = id.rawId(); }
    void SetDTDetId(const DTChamberId& id) { rawDetId = id.rawId(); }

    //CSCDetId CSC_DetId                          () const { return csc_DetId;    }
    //RPCDetId RPC_DetId                          () const { return rpc_DetId;    }
    //GEMDetId GEM_DetId                          () const { return gem_DetId;    }
    //CSCCorrelatedLCTDigi CSC_LCTDigi            () const { return csc_LCTDigi;  }
    //RPCDigi RPC_Digi                            () const { return rpc_Digi;     }
    //CPPFDigi CPPF_Digi                          () const { return cppf_Digi;    }
    //GEMPadDigiCluster GEM_PadClusterDigi        () const { return gem_PadClusterDigi;  }
    CSCDetId CSC_DetId() const { return CSCDetId(rawDetId); }
    RPCDetId RPC_DetId() const { return RPCDetId(rawDetId); }
    GEMDetId GEM_DetId() const { return GEMDetId(rawDetId); }
    ME0DetId ME0_DetId() const { return ME0DetId(rawDetId); }
    DTChamberId DT_DetId() const { return DTChamberId(rawDetId); }

    void set_subsystem(int bits) { subsystem = bits; }
    void set_endcap(int bits) { endcap = bits; }
    void set_station(int bits) { station = bits; }
    void set_ring(int bits) { ring = bits; }
    void set_sector(int bits) { sector = bits; }
    void set_sector_RPC(int bits) { sector_RPC = bits; }
    void set_sector_idx(int bits) { sector_idx = bits; }
    void set_subsector(int bits) { subsector = bits; }
    void set_subsector_RPC(int bits) { subsector_RPC = bits; }
    void set_chamber(int bits) { chamber = bits; }
    void set_csc_ID(int bits) { csc_ID = bits; }
    void set_csc_nID(int bits) { csc_nID = bits; }
    void set_roll(int bits) { roll = bits; }
    void set_neighbor(int bits) { neighbor = bits; }
    void set_mpc_link(int bits) { mpc_link = bits; }
    void set_pc_sector(int bits) { pc_sector = bits; }
    void set_pc_station(int bits) { pc_station = bits; }
    void set_pc_chamber(int bits) { pc_chamber = bits; }
    void set_pc_segment(int bits) { pc_segment = bits; }
    void set_wire(int bits) { wire = bits; }
    void set_strip(int bits) { strip = bits; }
    void set_strip_hi(int bits) { strip_hi = bits; }
    void set_strip_low(int bits) { strip_low = bits; }
    void set_strip_quart(int bits) { strip_quart = bits; }            // Run 3
    void set_strip_eighth(int bits) { strip_eighth = bits; }          // Run 3
    void set_strip_quart_bit(int bits) { strip_quart_bit = bits; }    // Run 3
    void set_strip_eighth_bit(int bits) { strip_eighth_bit = bits; }  // Run 3
    void set_track_num(int bits) { track_num = bits; }
    void set_quality(int bits) { quality = bits; }
    void set_pattern(int bits) { pattern = bits; }
    void set_pattern_run3(int bits) { pattern_run3 = bits; }  // Run 3
    void set_bend(int bits) { bend = bits; }
    void set_slope(int bits) { slope = bits; }  // Run 3
    void set_valid(int bits) { valid = bits; }
    void set_sync_err(int bits) { sync_err = bits; }
    // GEM specific aliases
    void set_pad(int bits) { set_strip(bits); }
    void set_pad_hi(int bits) { set_strip_hi(bits); }
    void set_pad_low(int bits) { set_strip_low(bits); }
    void set_partition(int bits) { set_roll(bits); }
    void set_layer(int bits) { layer = bits; }
    void set_cluster_size(int bits) { set_quality(bits); }
    void set_cluster_id(int bits) { set_track_num(bits); }
    // END GEM specific
    void set_bc0(int bits) { bc0 = bits; }
    void set_bx(int bits) { bx = bits; }
    void set_stub_num(int bits) { stub_num = bits; }
    void set_phi_fp(int bits) { phi_fp = bits; }
    void set_theta_fp(int bits) { theta_fp = bits; }
    void set_zone_hit(int bits) { zone_hit = bits; }
    void set_zone_code(int bits) { zone_code = bits; }
    void set_fs_segment(int bits) { fs_segment = bits; }
    void set_fs_zone_code(int bits) { fs_zone_code = bits; }
    void set_bt_station(int bits) { bt_station = bits; }
    void set_bt_segment(int bits) { bt_segment = bits; }
    void set_phi_loc(float val) { phi_loc = val; }
    void set_phi_glob(float val) { phi_glob = val; }
    void set_theta(float val) { theta = val; }
    void set_eta(float val) { eta = val; }
    void set_time(float val) { time = val; }
    void set_phi_sim(float val) { phi_sim = val; }
    void set_theta_sim(float val) { theta_sim = val; }
    void set_eta_sim(float val) { eta_sim = val; }
    void set_rho_sim(float val) { rho_sim = val; }
    void set_z_sim(float val) { z_sim = val; }
    void set_alct_quality(int bits) { alct_quality = bits; }
    void set_clct_quality(int bits) { clct_quality = bits; }

    int Subsystem() const { return subsystem; }
    int Endcap() const { return endcap; }
    int Station() const { return station; }
    int Ring() const { return ring; }
    int Sector() const { return sector; }
    int Sector_RPC() const { return sector_RPC; }
    int Sector_idx() const { return sector_idx; }
    int Subsector() const { return subsector; }
    int Subsector_RPC() const { return subsector_RPC; }
    int Chamber() const { return chamber; }
    int CSC_ID() const { return csc_ID; }
    int CSC_nID() const { return csc_nID; }
    int Roll() const { return roll; }
    int Neighbor() const { return neighbor; }
    int MPC_link() const { return mpc_link; }
    int PC_sector() const { return pc_sector; }
    int PC_station() const { return pc_station; }
    int PC_chamber() const { return pc_chamber; }
    int PC_segment() const { return pc_segment; }
    int Wire() const { return wire; }
    int Strip() const { return strip; }
    int Strip_hi() const { return strip_hi; }
    int Strip_low() const { return strip_low; }
    int Strip_quart() const { return strip_quart; }            // Run 3
    int Strip_eighth() const { return strip_eighth; }          // Run 3
    int Strip_quart_bit() const { return strip_quart_bit; }    // Run 3
    int Strip_eighth_bit() const { return strip_eighth_bit; }  // Run 3
    int Track_num() const { return track_num; }
    int Quality() const { return quality; }
    int Pattern() const { return pattern; }
    int Pattern_run3() const { return pattern_run3; }  // Run 3
    int Bend() const { return bend; }
    int Slope() const { return slope; }  // Run 3
    int Valid() const { return valid; }
    int Sync_err() const { return sync_err; }
    // GEM specific aliases for member variables that don't match GEM nomenclature
    /*
     * Each GEM pad is the OR of two neighbouring strips in phi.
     * For GE1/1 (10 degree chambers) this results in a total of 192 pads per eta partition
       * 128 strips per phi sector
       * 3 phi sectors per eta partition
     * For GE2/1 (20 degree chambers) this results in a total of 384 pads per eta partition
       * 128 strips per phi sector
       * 6 phi sectors per eta partition
     */
    /// Repurpose "strip" as GEM pad for GEM sourced hits
    int Pad() const { return Strip(); }
    /// Repurpose "strip" as GEM pad for GEM sourced hits
    int Pad_hi() const { return Strip_hi(); }
    /// Repurpose "strip" as GEM pad for GEM sourced hits
    int Pad_low() const { return Strip_low(); }
    /// "roll" corresponds to the GEM eta partition
    int Partition() const { return Roll(); }
    int Layer() const { return layer; }
    /// Repurpose "quality" as the GEM cluster_size (number of pads in the cluster)
    int ClusterSize() const { return Quality(); }
    /// Repurpose "track_num" as the GEM cluster_id
    int ClusterID() const { return Track_num(); }
    // END GEM specific
    int BC0() const { return bc0; }
    int BX() const { return bx; }
    int Stub_num() const { return stub_num; }
    int Phi_fp() const { return phi_fp; }
    int Theta_fp() const { return theta_fp; }
    int Zone_hit() const { return zone_hit; }
    int Zone_code() const { return zone_code; }
    int FS_segment() const { return fs_segment; }
    int FS_zone_code() const { return fs_zone_code; }
    int BT_station() const { return bt_station; }
    int BT_segment() const { return bt_segment; }
    float Phi_loc() const { return phi_loc; }
    float Phi_glob() const { return phi_glob; }
    float Theta() const { return theta; }
    float Eta() const { return eta; }
    float Time() const { return time; }
    float Phi_sim() const { return phi_sim; }
    float Theta_sim() const { return theta_sim; }
    float Eta_sim() const { return eta_sim; }
    float Rho_sim() const { return rho_sim; }
    float Z_sim() const { return z_sim; }
    int ALCT_quality() const { return alct_quality; }
    int CLCT_quality() const { return clct_quality; }

    bool Is_DT() const { return subsystem == l1tmu::kDT; }
    bool Is_CSC() const { return subsystem == l1tmu::kCSC; }
    bool Is_RPC() const { return subsystem == l1tmu::kRPC; }
    bool Is_GEM() const { return subsystem == l1tmu::kGEM; }
    bool Is_ME0() const { return subsystem == l1tmu::kME0; }

  private:
    //CSCDetId csc_DetId;
    //RPCDetId rpc_DetId;
    //GEMDetId gem_DetId;
    //CSCCorrelatedLCTDigi csc_LCTDigi;
    //RPCDigi rpc_Digi;
    //CPPFDigi cppf_Digi;
    //GEMPadDigiCluster gem_PadClusterDigi;

    uint32_t rawDetId;  ///< raw CMSSW DetId
    int subsystem;      ///<  0 -  4.  0 for DT, 1 for CSC, 2 for RPC, 3 for GEM, 4 for ME0
    int endcap;         ///<    +/-1.  For ME+ and ME-.
    int station;        ///<  1 -  4.
    int ring;  ///<  1 -  4.  ME1/1a is denoted as "Ring 4".  Should check dependence on input CSCDetId convention. - AWB 02.03.17
    int sector;      ///<  1 -  6.  CSC / GEM / EMTF sector convention: sector 1 starts at 15 degrees
    int sector_RPC;  ///<  1 -  6.  RPC sector convention (in CMSSW): sector 1 starts at -5 degrees
    int sector_idx;  ///<  0 - 11.  0 - 5 for ME+, 6 - 11 for ME-.  For neighbor hits, set by EMTF sector that received it.
    int subsector;  ///<  0 -  6.  In CSCs, 1 or 2 for ME1, 0 for ME2/3/4.
    int subsector_RPC;  ///<  0 -  6.  RPC sector convention (in CMSSW): subsector 3 is the first chamber in the EMTF sector.
    int chamber;        ///<  1 - 36.  Chamber 1 starts at -5 degrees.
    int csc_ID;         ///<  1 -  9.  For CSCs only.
    int csc_nID;        ///<  1 - 15.  For CSCs only.  Neighbors 10 - 15, 12 not filled.
    int roll;           ///<  1 -  3.  For RPCs only, sub-division of ring. (Range? - AWB 02.03.17)
    int neighbor;       ///<  0 or 1.  Filled in EMTFBlock(ME|GEM|RPC).cc
    int mpc_link;       ///<  1 -  3.  Filled in EMTFHit.cc from CSCCorrelatedLCTDigi
    int pc_sector;         ///<  1 -  6.  EMTF sector that received the LCT, even those sent from neighbor sectors.
    int pc_station;        ///<  0 -  5.  0 for ME1 subsector 1, 5 for neighbor hits.
    int pc_chamber;        ///<  0 -  8.
    int pc_segment;        ///<  0 -  3.
    int wire;              ///<  0 - 111  For CSCs only.
    int strip;             ///<  0 - 158  For CSCs only.
    int strip_hi;          ///<  ? -  ?.  For RPCs only, highest strip in a cluster.  (Range? - AWB 02.03.17)
    int strip_low;         ///<  ? -  ?.  For RPCs only, lowest strip in a cluster.  (Range? - AWB 02.03.17)
    int strip_quart;       ///< Run 3 CSC parameters
    int strip_eighth;      ///< Run 3 CSC parameters
    int strip_quart_bit;   ///< Run 3 CSC parameters
    int strip_eighth_bit;  ///< Run 3 CSC parameters
    int track_num;         ///<  ? -  ?.  For CSCs only.  (Range? - AWB 02.03.17)
    int quality;           ///<  0 - 15.  For CSCs only.
    int pattern;           ///<  0 - 10.  For CSCs only.
    int pattern_run3;      ///< Run 3 For CSC only.
    int bend;              ///<  0 or 1.  For CSCs only.
    int slope;             ///<  Run 3 For CSC only.
    int valid;             ///<  0 or 1.  For CSCs only (for now; could use to flag failing clusters? - AWB 02.03.17)
    int sync_err;          ///<  0 or 1.  For CSCs only.
    // GEM specific
    int layer;  ///<  0 -   1.  For GEMs only, superchamber detector layer (1 or 2).
    // END GEM specific
    int bc0;           ///<  0 or 1.  Only from unpacked data? - AWB 02.03.17
    int bx;            ///< -3 - +3.
    int stub_num;      ///<  0 or 1.  Only from unpacked data? - AWB 02.03.17
    int phi_fp;        ///<  0 - 4920
    int theta_fp;      ///<  0 - 127
    int zone_hit;      ///<  4 - 156  (Range? - AWB 02.03.17)
    int zone_code;     ///<  0 - 12.  (Range? - AWB 02.03.17)
    int fs_segment;    ///<  0 - 13.  (Range? - AWB 02.03.17)
    int fs_zone_code;  ///<  1 - 14.  (Range? - AWB 02.03.17)
    int bt_station;    ///<  0 -  4.
    int bt_segment;    ///<  0 - 25.  (Range? - AWB 02.03.17)
    float phi_loc;     ///< -20 - 60  (Range? - AWB 02.03.17)
    float phi_glob;    ///< +/-180.
    float theta;       ///< 0 - 90.
    float eta;         ///< +/-2.5.
    float time;        ///<  ? -  ?.  RPC time information (ns)
    float phi_sim;     ///< +/-180.
    float theta_sim;   ///< 0 - 90.
    float eta_sim;     ///< +/-2.5.
    float rho_sim;     ///<  ? -  ?.
    float z_sim;       ///<  ? -  ?.
    int alct_quality;  ///<  1 -  3.  For emulated CSC LCTs only, maps to number of ALCT layers (4 - 6).
    int clct_quality;  ///<  4 -  6.  For emulated CSC LCTs only, maps to number of CLCT layers (4 - 6).

  };  // End of class EMTFHit

  // Define a vector of EMTFHit
  typedef std::vector<EMTFHit> EMTFHitCollection;

}  // End of namespace l1t

#endif /* define DataFormats_L1TMuon_EMTFHit_h */
