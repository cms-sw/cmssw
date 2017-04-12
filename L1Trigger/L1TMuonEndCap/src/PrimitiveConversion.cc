#include "L1Trigger/L1TMuonEndCap/interface/PrimitiveConversion.hh"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "L1Trigger/L1TMuonEndCap/interface/SectorProcessorLUT.hh"
#include "L1Trigger/L1TMuonEndCap/interface/TrackTools.hh"

using CSCData = L1TMuon::TriggerPrimitive::CSCData;
using RPCData = L1TMuon::TriggerPrimitive::RPCData;


void PrimitiveConversion::configure(
    const L1TMuon::GeometryTranslator* tp_geom,
    const SectorProcessorLUT* lut,
    int verbose, int endcap, int sector, int bx,
    int bxShiftCSC, int bxShiftRPC,
    const std::vector<int>& zoneBoundaries, int zoneOverlap, int zoneOverlapRPC,
    bool duplicateTheta, bool fixZonePhi, bool useNewZones, bool fixME11Edges,
    bool bugME11Dupes
) {
  assert(tp_geom != nullptr);
  assert(lut != nullptr);

  tp_geom_ = tp_geom;
  lut_     = lut;

  verbose_ = verbose;
  endcap_  = endcap;
  sector_  = sector;
  bx_      = bx;

  bxShiftCSC_      = bxShiftCSC;
  bxShiftRPC_      = bxShiftRPC;

  zoneBoundaries_  = zoneBoundaries;
  zoneOverlap_     = zoneOverlap;
  zoneOverlapRPC_  = zoneOverlapRPC;
  duplicateTheta_  = duplicateTheta;
  fixZonePhi_      = fixZonePhi;
  useNewZones_     = useNewZones;
  fixME11Edges_    = fixME11Edges;
  bugME11Dupes_    = bugME11Dupes;
}


// Specialized for CSC
template<>
void PrimitiveConversion::process(
    CSCTag tag,
    const std::map<int, TriggerPrimitiveCollection>& selected_csc_map,
    EMTFHitCollection& conv_hits
) const {
  std::map<int, TriggerPrimitiveCollection>::const_iterator map_tp_it  = selected_csc_map.begin();
  std::map<int, TriggerPrimitiveCollection>::const_iterator map_tp_end = selected_csc_map.end();

  for (; map_tp_it != map_tp_end; ++map_tp_it) {
    // Unique chamber ID in FW, {0, 53} as defined in get_index_csc in src/PrimitiveSelection.cc
    int selected   = map_tp_it->first;
    // "Primitive Conversion" sector/station/chamber ID scheme used in FW
    int pc_sector  = sector_;
    int pc_station = selected / 9;  // {0, 5} = {ME1 sub 1, ME1 sub 2, ME2, ME3, ME4, neighbor}
    int pc_chamber = selected % 9;  // Equals CSC ID - 1 for all except neighbor chambers
    int pc_segment = 0;             // Counts hits in a single chamber

    TriggerPrimitiveCollection::const_iterator tp_it  = map_tp_it->second.begin();
    TriggerPrimitiveCollection::const_iterator tp_end = map_tp_it->second.end();

    for (; tp_it != tp_end; ++tp_it) {
      EMTFHit conv_hit;
      convert_csc(pc_sector, pc_station, pc_chamber, pc_segment, *tp_it, conv_hit);
      conv_hits.push_back(conv_hit);
      pc_segment += 1;
    }

    assert(pc_segment <= 4);  // With 2 unique LCTs, 4 possible strip/wire combinations
  }
}

// Specialized for RPC
template<>
void PrimitiveConversion::process(
    RPCTag tag,
    const std::map<int, TriggerPrimitiveCollection>& selected_rpc_map,
    EMTFHitCollection& conv_hits
) const {
  std::map<int, TriggerPrimitiveCollection>::const_iterator map_tp_it  = selected_rpc_map.begin();
  std::map<int, TriggerPrimitiveCollection>::const_iterator map_tp_end = selected_rpc_map.end();

  for (; map_tp_it != map_tp_end; ++map_tp_it) {
    int selected   = map_tp_it->first;
    // Try to match the definitions used for CSC primitive conversion (unconfirmed!)
    int pc_sector  = sector_;
    int pc_station = (selected < 12 ? (selected / 6) + 1 : (selected / 12) + 2);  // {1, 5} = {RE1, RE2, RE3, RE4, neighbor}
    if (pc_station == 1)  pc_station = 0;  // because CSC pc_station 0 has neighbor, but pc_station 1 has no neighbor
    int pc_chamber = (selected < 12 ? (selected % 6) : (selected % 12));  // Unique identifier per station
    int pc_segment = 0;

    TriggerPrimitiveCollection::const_iterator tp_it  = map_tp_it->second.begin();
    TriggerPrimitiveCollection::const_iterator tp_end = map_tp_it->second.end();

    for (; tp_it != tp_end; ++tp_it) {
      EMTFHit conv_hit;
      convert_rpc(pc_sector, pc_station, pc_chamber, pc_segment, *tp_it, conv_hit);
      conv_hits.push_back(conv_hit);
      pc_segment += 1;
    }

    assert(pc_segment <= 2);  // at most 2 clusters per RPC chamber
  }
}

const SectorProcessorLUT& PrimitiveConversion::lut() const {
  return *lut_;
}

// _____________________________________________________________________________
// CSC functions
void PrimitiveConversion::convert_csc(
    int pc_sector, int pc_station, int pc_chamber, int pc_segment,
    const TriggerPrimitive& muon_primitive,
    EMTFHit& conv_hit
) const {
  const CSCDetId& tp_detId = muon_primitive.detId<CSCDetId>();
  const CSCData&  tp_data  = muon_primitive.getCSCData();

  int tp_endcap    = tp_detId.endcap();
  int tp_sector    = tp_detId.triggerSector();
  int tp_station   = tp_detId.station();
  int tp_ring      = tp_detId.ring();
  int tp_chamber   = tp_detId.chamber();

  int tp_bx        = tp_data.bx;
  int tp_csc_ID    = tp_data.cscID;

  // station 1 --> subsector 1 or 2
  // station 2,3,4 --> subsector 0
  int tp_subsector = (tp_station != 1) ? 0 : ((tp_chamber % 6 > 2) ? 1 : 2);

  const bool is_neighbor = (pc_station == 5);

  int csc_nID      = tp_csc_ID;  // modify csc_ID if coming from neighbor sector
  if (is_neighbor) {
    // station 1 has 3 neighbor chambers: 13,14,15 in rings 1,2,3
    // (where are chambers 10,11,12 in station 1? they were used to label ME1/1a, but not anymore)
    // station 2,3,4 have 2 neighbor chambers: 10,11 in rings 1,2
    csc_nID = (pc_chamber < 3) ? (pc_chamber + 12) : ( ((pc_chamber - 1) % 2) + 9);
    csc_nID += 1;

    if (tp_station == 1) {  // neighbor ME1
      assert(tp_subsector == 2);
    }
  }

  // Set properties
  conv_hit.SetCSCDetId     ( tp_detId );

  conv_hit.set_endcap      ( (tp_endcap == 2) ? -1 : tp_endcap );
  conv_hit.set_station     ( tp_station );
  conv_hit.set_ring        ( tp_ring );
  //conv_hit.set_roll        ( tp_roll );
  conv_hit.set_chamber     ( tp_chamber );
  conv_hit.set_sector      ( tp_sector );
  conv_hit.set_subsector   ( tp_subsector );
  conv_hit.set_csc_ID      ( tp_csc_ID );
  conv_hit.set_csc_nID     ( csc_nID );
  conv_hit.set_track_num   ( tp_data.trknmb );
  conv_hit.set_sync_err    ( tp_data.syncErr );

  conv_hit.set_bx          ( tp_bx + bxShiftCSC_ );
  conv_hit.set_subsystem   ( TriggerPrimitive::kCSC );
  conv_hit.set_is_CSC      ( true );
  conv_hit.set_is_RPC      ( false );

  conv_hit.set_pc_sector   ( pc_sector );
  conv_hit.set_pc_station  ( pc_station );
  conv_hit.set_pc_chamber  ( pc_chamber );
  conv_hit.set_pc_segment  ( pc_segment );

  conv_hit.set_valid       ( tp_data.valid );
  conv_hit.set_strip       ( tp_data.strip );
  //conv_hit.set_strip_low   ( tp_data.strip_low );
  //conv_hit.set_strip_hi    ( tp_data.strip_hi );
  conv_hit.set_wire        ( tp_data.keywire );
  conv_hit.set_quality     ( tp_data.quality );
  conv_hit.set_pattern     ( tp_data.pattern );
  conv_hit.set_bend        ( tp_data.bend );

  conv_hit.set_neighbor    ( is_neighbor );
  conv_hit.set_sector_idx  ( (endcap_ == 1) ? sector_ - 1 : sector_ + 5 );

  convert_csc_details(conv_hit);

  // Add coordinates from fullsim
  {
    const GlobalPoint& gp = tp_geom_->getGlobalPoint(muon_primitive);
    double glob_phi   = emtf::rad_to_deg(gp.phi().value());
    double glob_theta = emtf::rad_to_deg(gp.theta());
    double glob_eta   = gp.eta();

    conv_hit.set_phi_sim   ( glob_phi );
    conv_hit.set_theta_sim ( glob_theta );
    conv_hit.set_eta_sim   ( glob_eta );
  }
}

void PrimitiveConversion::convert_csc_details(EMTFHit& conv_hit) const {
  const bool is_neighbor = conv_hit.Neighbor();

  // Defined as in firmware
  // endcap : 0-1 for ME+,ME-
  // sector : 0-5
  // station: 0-4 for st1 sub1 or st1 sub2 from neighbor, st1 sub2, st2, st3, st4
  // cscid  : 0-14 (excluding 11), including neighbors
  const int fw_endcap  = (endcap_-1);
  const int fw_sector  = (sector_-1);
  const int fw_station = (conv_hit.Station() == 1) ? (is_neighbor ? 0 : (conv_hit.Subsector()-1)) : conv_hit.Station();
  const int fw_cscid   = (conv_hit.CSC_nID()-1);
  const int fw_strip   = conv_hit.Strip();  // it is half-strip, despite the name
  const int fw_wire    = conv_hit.Wire();   // it is wiregroup, despite the name

  // Primitive converter unit
  // station: 0-5 for st1 sub1, st1 sub2, st2, st3, st4, neigh all st*
  // chamber: 0-8
  const int pc_station = conv_hit.PC_station();
  const int pc_chamber = conv_hit.PC_chamber();

  const bool is_me11a = (conv_hit.Station() == 1 && conv_hit.Ring() == 4);
  const bool is_me11b = (conv_hit.Station() == 1 && conv_hit.Ring() == 1);
  const bool is_me13  = (conv_hit.Station() == 1 && conv_hit.Ring() == 3);

  // Is this chamber mounted in reverse direction?
  // (i.e., phi vs. strip number is reversed)
  bool ph_reverse = false;
  if ((fw_endcap == 0 && fw_station >= 3) || (fw_endcap == 1 && fw_station < 3))
    ph_reverse = true;

  // Chamber coverage if phi_reverse = true
  int ph_coverage = 0; // Offset for coordinate conversion
  if (ph_reverse) {
    if (fw_station <= 1 && ((fw_cscid >= 6 && fw_cscid <= 8) || fw_cscid == 14))  // ME1/3
      ph_coverage = 15;
    else if (fw_station >= 2 && (fw_cscid <= 2 || fw_cscid == 9))  // ME2,3,4/1
      ph_coverage = 40;
    else  // all others
      ph_coverage = 20;
  }

  // Is this 10-deg or 20-deg chamber?
  bool is_10degree = false;
  if (
      (fw_station <= 1) || // ME1
      (fw_station >= 2 && ((fw_cscid >= 3 && fw_cscid <= 8) || fw_cscid == 10))  // ME2,3,4/2
  ) {
    is_10degree = true;
  }

  // LUT index
  // There are 54 CSC chambers including the neighbors in a sector, but 61 LUT indices
  // This comes from dividing the 6 chambers + 1 neighbor in ME1/1 into ME1/1a and ME1/1b
  int pc_lut_id = pc_chamber;
  if (pc_station == 0) {         // ME1 sub 1: 0 - 11
    pc_lut_id = is_me11a ? pc_lut_id + 9 : pc_lut_id;
  } else if (pc_station == 1) {  // ME1 sub 2: 16 - 27
    pc_lut_id += 16;
    pc_lut_id = is_me11a ? pc_lut_id + 9 : pc_lut_id;
  } else if (pc_station == 2) {  // ME2: 28 - 36
    pc_lut_id += 28;
  } else if (pc_station == 3) {  // ME3: 39 - 47
    pc_lut_id += 39;
  } else if (pc_station == 4) {  // ME4 : 50 - 58
    pc_lut_id += 50;
  } else if (pc_station == 5 && pc_chamber < 3) {  // neighbor ME1: 12 - 15
    pc_lut_id = is_me11a ? pc_lut_id + 15 : pc_lut_id + 12;
  } else if (pc_station == 5 && pc_chamber < 5) {  // neighbor ME2: 37 - 38
    pc_lut_id += 28 + 9 - 3;
  } else if (pc_station == 5 && pc_chamber < 7) {  // neighbor ME3: 48 - 49
    pc_lut_id += 39 + 9 - 5;
  } else if (pc_station == 5 && pc_chamber < 9) {  // neighbor ME4: 59 - 60
    pc_lut_id += 50 + 9 - 7;
  }
  assert(pc_lut_id < 61);

  if (verbose_ > 1) {  // debug
    std::cout << "pc_station: " << pc_station << " pc_chamber: " << pc_chamber
        << " fw_station: " << fw_station << " fw_cscid: " << fw_cscid
        << " lut_id: " << pc_lut_id
        << " ph_init: " << lut().get_ph_init(fw_endcap, fw_sector, pc_lut_id)
        << " ph_disp: " << lut().get_ph_disp(fw_endcap, fw_sector, pc_lut_id)
        << " th_init: " << lut().get_th_init(fw_endcap, fw_sector, pc_lut_id)
        << " th_disp: " << lut().get_th_disp(fw_endcap, fw_sector, pc_lut_id)
        << " ph_init_hard: " << lut().get_ph_init_hard(fw_station, fw_cscid)
        << std::endl;
  }

  // ___________________________________________________________________________
  // phi conversion

  // Convert half-strip into 1/8-strip
  int eighth_strip = 0;

  // Apply phi correction from CLCT pattern number (from src/SectorProcessorLUT.cc)
  int clct_pat_corr = lut().get_ph_patt_corr(conv_hit.Pattern());
  int clct_pat_corr_sign = (lut().get_ph_patt_corr_sign(conv_hit.Pattern()) == 0) ? 1 : -1;

  // At strip number 0, protect against negative correction
  bool bugStrip0BeforeFW48200 = false;
  if (bugStrip0BeforeFW48200 == false && fw_strip == 0 && clct_pat_corr_sign == -1)
    clct_pat_corr = 0;

  if (is_10degree) {
    eighth_strip = fw_strip << 2;  // full precision, uses only 2 bits of pattern correction
    eighth_strip += clct_pat_corr_sign * (clct_pat_corr >> 1);
  } else {
    eighth_strip = fw_strip << 3;  // multiply by 2, uses all 3 bits of pattern correction
    eighth_strip += clct_pat_corr_sign * (clct_pat_corr >> 0);
  }
  assert(bugStrip0BeforeFW48200 == true || eighth_strip >= 0);

  // Multiplicative factor for eighth_strip
  int factor = 1024;
  if (is_me11a)
    factor = 1707;  // ME1/1a
  else if (is_me11b)
    factor = 1301;  // ME1/1b
  else if (is_me13)
    factor = 947;   // ME1/3

  // ph_tmp is full-precision phi, but local to chamber (counted from strip 0)
  // full phi precision: ( 1/60) deg (1/8-strip)
  // zone phi precision: (32/60) deg (4-strip, 32 times coarser than full phi precision)
  int ph_tmp = (eighth_strip * factor) >> 10;
  int ph_tmp_sign = (ph_reverse == 0) ? 1 : -1;

  int fph = lut().get_ph_init(fw_endcap, fw_sector, pc_lut_id);
  fph = fph + ph_tmp_sign * ph_tmp;

  // // Add option to correct for endcap-dependence (needed to line up with global geometry / RPC hits) - AWB 03.03.17
  // fph -= ( (fw_endcap == 0) ? 28 : 36);

  int ph_hit = lut().get_ph_disp(fw_endcap, fw_sector, pc_lut_id);
  ph_hit = (ph_hit >> 1) + ph_tmp_sign * (ph_tmp >> 5) + ph_coverage;

  // Full phi +16 to put the rounded value into the middle of error range
  // Divide full phi by 32, subtract chamber start
  int ph_hit_fixed = -1 * lut().get_ph_init_hard(fw_station, fw_cscid);
  ph_hit_fixed = ph_hit_fixed + ((fph + (1<<4)) >> 5);

  if (fixZonePhi_)
    ph_hit = ph_hit_fixed;

  // Zone phi
  int zone_hit = lut().get_ph_zone_offset(pc_station, pc_chamber);
  zone_hit += ph_hit;

  int zone_hit_fixed = lut().get_ph_init_hard(fw_station, fw_cscid);
  zone_hit_fixed += ph_hit_fixed;
  // Since ph_hit_fixed = ((fph + (1<<4)) >> 5) - lut().get_ph_init_hard(), the following is equivalent:
  //zone_hit_fixed = ((fph + (1<<4)) >> 5);

  if (fixZonePhi_)
    zone_hit = zone_hit_fixed;

  assert(0 <= fph && fph < 5000);
  assert(0 <= zone_hit && zone_hit < 192);

  // ___________________________________________________________________________
  // theta conversion

  // th_tmp is theta local to chamber
  int pc_wire_id = (fw_wire & 0x7f);  // 7-bit
  int th_tmp = lut().get_th_lut(fw_endcap, fw_sector, pc_lut_id, pc_wire_id);

  // For ME1/1 with tilted wires, add theta correction as a function of (wire,strip) index
  if (!fixME11Edges_ && (is_me11a || is_me11b)) {
    int pc_wire_strip_id = (((fw_wire >> 4) & 0x3) << 5) | ((eighth_strip >> 4) & 0x1f);  // 2-bit from wire, 5-bit from 2-strip

    // Only affect runs before FW changeset 47114 is applied
    // e.g. Run 281707 and earlier
    if (bugME11Dupes_) {
      bool bugME11DupesBeforeFW47114 = false;
      if (bugME11DupesBeforeFW47114) {
        if (conv_hit.PC_segment() == 1) {
          pc_wire_strip_id = (((fw_wire >> 4) & 0x3) << 5) | (0);  // 2-bit from wire, 5-bit from 2-strip
        }
      }
    }

    int th_corr = lut().get_th_corr_lut(fw_endcap, fw_sector, pc_lut_id, pc_wire_strip_id);
    int th_corr_sign = (ph_reverse == 0) ? 1 : -1;

    th_tmp = th_tmp + th_corr_sign * th_corr;

    // Check that correction did not make invalid value outside chamber coverage
    const int th_negative = 50;
    const int th_coverage = 45;
    if (th_tmp > th_negative || th_tmp < 0 || fw_wire == 0)
      th_tmp = 0;  // limit at the bottom
    if (th_tmp > th_coverage)
      th_tmp = th_coverage;  // limit at the top

  } else if (fixME11Edges_ && (is_me11a || is_me11b)) {
    int pc_wire_strip_id = (((fw_wire >> 4) & 0x3) << 5) | ((eighth_strip >> 4) & 0x1f);  // 2-bit from wire, 5-bit from 2-strip
    if (is_me11a)
      pc_wire_strip_id = (((fw_wire >> 4) & 0x3) << 5) | ((((eighth_strip*341)>>8) >> 4) & 0x1f);  // correct for ME1/1a strip number (341/256 =~ 1.333)
    int th_corr = lut().get_th_corr_lut(fw_endcap, fw_sector, pc_lut_id, pc_wire_strip_id);

    th_tmp = th_tmp + th_corr;

    // Check that correction did not make invalid value outside chamber coverage
    const int th_coverage = 46;  // max coverage for front chamber is 47, max coverage for rear chamber is 45
    if (fw_wire == 0)
      th_tmp = 0;  // limit at the bottom
    if (th_tmp > th_coverage)
      th_tmp = th_coverage;  // limit at the top
  }

  // theta precision: (36.5/128) deg
  // theta starts at 8.5 deg: {1, 127} <--> {8.785, 44.715}
  int th = lut().get_th_init(fw_endcap, fw_sector, pc_lut_id);
  th = th + th_tmp;

  assert(0 <=  th &&  th <  128);
  th = (th == 0) ? 1 : th;  // protect against invalid value


  // ___________________________________________________________________________
  // zones

  static const unsigned int zone_code_table[4][3] = {  // map (station,ring) to zone_code
    {0b0011, 0b0100, 0b1000},  // st1 r1: [z0,z1], r2: [z2],      r3: [z3]
    {0b0011, 0b1100, 0b0000},  // st2 r1: [z0,z1], r2: [z2,z3]
    {0b0001, 0b1110, 0b0000},  // st3 r1: [z0],    r2: [z1,z2,z3]
    {0b0001, 0b0110, 0b0000}   // st4 r1: [z0],    r2: [z1,z2]
  };

  static const unsigned int zone_code_table_new[4][3] = {  // map (station,ring) to zone_code
    {0b0011, 0b0110, 0b1000},  // st1 r1: [z0,z1], r2: [z1,z2],   r3: [z3]
    {0b0011, 0b1110, 0b0000},  // st2 r1: [z0,z1], r2: [z1,z2,z3]
    {0b0011, 0b1110, 0b0000},  // st3 r1: [z0,z1], r2: [z1,z2,z3]
    {0b0001, 0b0110, 0b0000}   // st4 r1: [z0],    r2: [z1,z2]
  };

  struct {
    constexpr unsigned int operator()(int tp_station, int tp_ring, bool use_new_table) {
      unsigned int istation = (tp_station-1);
      unsigned int iring = (tp_ring == 4) ? 0 : (tp_ring-1);
      assert(istation < 4 && iring < 3);
      unsigned int zone_code = (use_new_table) ? zone_code_table_new[istation][iring] : zone_code_table[istation][iring];
      return zone_code;
    }
  } zone_code_func;

  // ph zone boundaries for chambers that cover more than one zone
  // bnd1 is the lower boundary, bnd2 the upper boundary
  int zone_code = 0;
  for (int izone = 0; izone < NUM_ZONES; ++izone) {
    int zone_code_tmp = zone_code_func(conv_hit.Station(), conv_hit.Ring(), useNewZones_);
    if (zone_code_tmp & (1<<izone)) {
      bool no_use_bnd1 = ((izone==0) || ((zone_code_tmp & (1<<(izone-1))) == 0) || is_me13);  // first possible zone for this hit
      bool no_use_bnd2 = (((zone_code_tmp & (1<<(izone+1))) == 0) || is_me13);  // last possible zone for this hit

      int ph_zone_bnd1 = no_use_bnd1 ? zoneBoundaries_.at(0) : zoneBoundaries_.at(izone);
      int ph_zone_bnd2 = no_use_bnd2 ? zoneBoundaries_.at(NUM_ZONES) : zoneBoundaries_.at(izone+1);
      int zone_overlap = zoneOverlap_;

      if ((th > (ph_zone_bnd1 - zone_overlap)) && (th <= (ph_zone_bnd2 + zone_overlap))) {
        zone_code |= (1<<izone);
      }
    }
  }
  assert(zone_code > 0);

  // For backward compatibility, no longer needed (only explicitly used in FW)
  // phzvl: each chamber overlaps with at most 3 zones, so this "local" zone word says
  // which of the possible zones contain the hit: 1 for lower, 2 for middle, 4 for upper
  int phzvl = 0;
  if (conv_hit.Ring() == 1 || conv_hit.Ring() == 4) {
    phzvl = (zone_code >> 0);
  } else if (conv_hit.Ring() == 2) {
    if (conv_hit.Station() == 3 || conv_hit.Station() == 4) {
      phzvl = (zone_code >> 1);
    } else if (conv_hit.Station() == 1 || conv_hit.Station() == 2) {
      phzvl = (zone_code >> 2);
    }
  } else if (conv_hit.Ring() == 3) {
    phzvl = (zone_code >> 3);
  }

  // ___________________________________________________________________________
  // For later use in primitive matching
  // (in the firmware, this happens in the find_segment module)

  int fs_history = 0;                       // history id: not set here, to be set in primitive matching
  int fs_chamber = -1;                      // chamber id
  int fs_segment = conv_hit.PC_segment() % 2; // segment id
  int fs_zone_code = zone_code_func(conv_hit.Station(), conv_hit.Ring(), useNewZones_);

  // For ME1
  //   j = 0 is neighbor sector chamber
  //   j = 1,2,3 are native subsector 1 chambers
  //   j = 4,5,6 are native subsector 2 chambers
  // For ME2,3,4:
  //   j = 0 is neighbor sector chamber
  //   j = 1,2,3,4,5,6 are native sector chambers
  if (fw_station <= 1) {  // ME1
    int n = (conv_hit.CSC_ID()-1) % 3;
    fs_chamber = is_neighbor ? 0 : ((conv_hit.Subsector() == 1) ? 1+n : 4+n);
  } else {  // ME2,3,4
    int n = (conv_hit.Ring() == 1) ? (conv_hit.CSC_ID()-1) : (conv_hit.CSC_ID()-1-3);
    fs_chamber = is_neighbor ? 0 : 1+n;
  }

  assert(fs_history >= 0 && fs_chamber != -1 && fs_segment < 2);
  // fs_segment is a 6-bit word, HHCCCS, encoding the segment number S in the chamber (1 or 2),
  // the chamber number CCC ("j" above: uniquely identifies chamber within station and ring),
  // and the history HH (0 for current BX, 1 for previous BX, 2 for BX before that)
  fs_segment = ((fs_history & 0x3)<<4) | ((fs_chamber & 0x7)<<1) | (fs_segment & 0x1);

  // ___________________________________________________________________________
  // For later use in angle calculation and best track selection
  // (in the firmware, this happens in the best_tracks module)

  int bt_station = fw_station;
  int bt_history = 0;
  int bt_chamber = fw_cscid+1;
  if (fw_station == 0 && bt_chamber >= 13)  // ME1 neighbor chambers 13,14,15 -> 10,11,12
    bt_chamber -= 3;
  int bt_segment = conv_hit.PC_segment() % 2;

  bt_segment = ((bt_history & 0x3)<<5) | ((bt_chamber & 0xf)<<1) | (bt_segment & 0x1);

  // ___________________________________________________________________________
  // Output

  conv_hit.set_phi_fp     ( fph );        // Full-precision integer phi
  conv_hit.set_theta_fp   ( th );         // Full-precision integer theta
  conv_hit.set_phzvl      ( phzvl );      // Local zone word: (1*low) + (2*mid) + (4*low) - used in FW debugging
  conv_hit.set_ph_hit     ( ph_hit );     // Intermediate quantity in phi calculation - used in FW debugging
  conv_hit.set_zone_hit   ( zone_hit );   // Phi value for building patterns (0.53333 deg precision)
  conv_hit.set_zone_code  ( zone_code );  // Full zone word: 1*(zone 0) + 2*(zone 1) + 4*(zone 2) + 8*(zone 3)

  conv_hit.set_fs_segment   ( fs_segment );    // Segment number used in primitive matching
  conv_hit.set_fs_zone_code ( fs_zone_code );  // Zone word used in primitive matching

  conv_hit.set_bt_station   ( bt_station );
  conv_hit.set_bt_segment   ( bt_segment );

  conv_hit.set_phi_loc  ( emtf::calc_phi_loc_deg(fph) );
  conv_hit.set_phi_glob ( emtf::calc_phi_glob_deg(conv_hit.Phi_loc(), conv_hit.Sector()) );
  conv_hit.set_theta    ( emtf::calc_theta_deg_from_int(th) );
  conv_hit.set_eta      ( emtf::calc_eta_from_theta_deg(conv_hit.Theta(), conv_hit.Endcap()) );
}

// _____________________________________________________________________________
// RPC functions
void PrimitiveConversion::convert_rpc(
    int pc_sector, int pc_station, int pc_chamber, int pc_segment,
    const TriggerPrimitive& muon_primitive,
    EMTFHit& conv_hit
) const {
  const RPCDetId& tp_detId = muon_primitive.detId<RPCDetId>();
  const RPCData&  tp_data  = muon_primitive.getRPCData();

  int tp_region    = tp_detId.region();     // 0 for Barrel, +/-1 for +/- Endcap
  int tp_endcap    = (tp_region == -1) ? 2 : tp_region;
  int tp_sector    = tp_detId.sector();     // 1 - 6 (60 degrees in phi, sector 1 begins at -5 deg)
  int tp_subsector = tp_detId.subsector();  // 1 - 6 (10 degrees in phi; staggered in z)
  int tp_station   = tp_detId.station();    // 1 - 4
  int tp_ring      = tp_detId.ring();       // 2 - 3 (increasing theta)
  int tp_roll      = tp_detId.roll();       // 1 - 3 (decreasing theta; aka A - C; space between rolls is 9 - 15 in theta_fp)

  int tp_bx        = tp_data.bx;

  const bool is_neighbor = (pc_station == 5);

  // Set properties
  conv_hit.SetRPCDetId     ( tp_detId );

  conv_hit.set_endcap      ( (tp_endcap == 2) ? -1 : tp_endcap );
  conv_hit.set_station     ( tp_station );
  conv_hit.set_ring        ( tp_ring );
  conv_hit.set_roll        ( tp_roll );
  //conv_hit.set_chamber     ( tp_chamber );
  conv_hit.set_sector      ( tp_sector );
  conv_hit.set_subsector   ( tp_subsector );
  //conv_hit.set_csc_ID      ( tp_csc_ID );
  //conv_hit.set_csc_nID     ( csc_nID );
  //conv_hit.set_track_num   ( tp_data.trknmb );
  //conv_hit.set_sync_err    ( tp_data.syncErr );

  conv_hit.set_bx          ( tp_bx + bxShiftRPC_ );
  conv_hit.set_subsystem   ( TriggerPrimitive::kRPC );
  conv_hit.set_is_CSC      ( false );
  conv_hit.set_is_RPC      ( true );

  conv_hit.set_pc_sector   ( pc_sector );
  conv_hit.set_pc_station  ( pc_station );
  conv_hit.set_pc_chamber  ( pc_chamber );
  conv_hit.set_pc_segment  ( pc_segment );

  conv_hit.set_valid       ( true );
  conv_hit.set_strip       ( int((tp_data.strip_low + tp_data.strip_hi) / 2) );  // in full-strip unit
  conv_hit.set_strip_low   ( tp_data.strip_low );
  conv_hit.set_strip_hi    ( tp_data.strip_hi );
  //conv_hit.set_wire        ( tp_data.keywire );
  //conv_hit.set_quality     ( tp_data.quality );
  conv_hit.set_pattern     ( 10 );  // Arbitrarily set to the straightest pattern for RPC hits
  //conv_hit.set_bend        ( tp_data.bend );

  conv_hit.set_neighbor    ( is_neighbor );
  conv_hit.set_sector_idx  ( (endcap_ == 1) ? sector_ - 1 : sector_ + 5 );


  // Get coordinates from fullsim since LUTs do not exist yet
  bool use_fullsim_coords = true;
  if (use_fullsim_coords) {
    const GlobalPoint& gp = tp_geom_->getGlobalPoint(muon_primitive);
    double glob_phi   = emtf::rad_to_deg(gp.phi().value());
    double glob_theta = emtf::rad_to_deg(gp.theta());
    double glob_eta   = gp.eta();

    int phi_loc_int   = emtf::calc_phi_loc_int(glob_phi, conv_hit.Sector());
    int theta_int     = emtf::calc_theta_int(glob_theta, conv_hit.Endcap());

    // Use RPC-specific convention in docs/CPPF-EMTF-format_2016_11_01.docx
    // Phi precision is (1/15) degrees, 4x larger than CSC precision of (1/60) degrees
    // Theta precision is (36.5/32) degrees, 4x larger than CSC precision of (36.5/128) degrees
    int fph = ((phi_loc_int + (1<<1)) >> 2) << 2;
    int th  = ((theta_int + (1<<1)) >> 2) << 2;
    assert(0 <= fph && fph < 4920);
    assert(0 <=  th &&  th <  128);
    th = (th == 0) ? 1 : th;  // protect against invalid value

    // _________________________________________________________________________
    // Output

    conv_hit.set_phi_sim   ( glob_phi );
    conv_hit.set_theta_sim ( glob_theta );
    conv_hit.set_eta_sim   ( glob_eta );

    conv_hit.set_phi_fp    ( fph ); // Full-precision integer phi
    conv_hit.set_theta_fp  ( th );  // Full-precision integer theta
  }

  convert_rpc_details(conv_hit);
}

void PrimitiveConversion::convert_rpc_details(EMTFHit& conv_hit) const {
  const bool is_neighbor = conv_hit.Neighbor();

  //const int fw_endcap  = (endcap_-1);
  //const int fw_sector  = (sector_-1);
  const int fw_station = (conv_hit.Station() == 1) ? 0 : conv_hit.Station();

  int fph = conv_hit.Phi_fp();
  int th  = conv_hit.Theta_fp();

  int zone_hit = ((fph + (1<<4)) >> 5);

  // ___________________________________________________________________________
  // zones

  // Compute the zone code based only on theta, with wider overlap (unconfirmed!)
  int zone_code = 0;
  for (int izone = 0; izone < NUM_ZONES; ++izone) {
    if (
        (th >  (zoneBoundaries_.at(izone)   - zoneOverlapRPC_)) &&
        (th <= (zoneBoundaries_.at(izone+1) + zoneOverlapRPC_))
    ) {
      zone_code |= (1 << izone);
    }
  }
  assert(zone_code > 0);

  // ___________________________________________________________________________
  // For later use in primitive matching (unconfirmed!)

  int fs_history = 0;                       // history id: not set here, to be set in primitive matching
  int fs_chamber = -1;                      // chamber id
  int fs_segment = conv_hit.PC_segment() % 2; // segment id
  int fs_zone_code = zone_code;             // same as zone code for now

  // For all RPC stations (REX)
  //   j = 0 is neighbor sector subsector
  //   j = 1,2,3,4,5,6 are native subsectors
  fs_chamber = is_neighbor ? 0 : ((conv_hit.Subsector() + 3) % 6) + 1;

  assert(fs_history >= 0 && fs_chamber != -1 && fs_segment < 2);
  // fs_segment is a 6-bit word, HHCCCS, encoding the segment number S in the subsector (0 or 1),
  // the the subsector number CCC ("j" above: uniquely identifies subsector within station and ring),
  // and the history HH (0 for current BX, 1 for previous BX, 2 for BX before that)
  fs_segment = ((fs_history & 0x3)<<4) | ((fs_chamber & 0x7)<<1) | (fs_segment & 0x1);

  // ___________________________________________________________________________
  // For later use in angle calculation and best track selection (unconfirmed!)

  int bt_station = fw_station;
  int bt_history = 0;
  int bt_chamber = 0;                       // wait for FW implementation
  int bt_segment = conv_hit.PC_segment() % 2;

  bt_segment = ((bt_history & 0x3)<<5) | ((bt_chamber & 0xf)<<1) | (bt_segment & 0x1);

  // ___________________________________________________________________________
  // Output

  conv_hit.set_phi_fp     ( fph );        // Full-precision integer phi
  conv_hit.set_theta_fp   ( th );         // Full-precision integer theta
  //conv_hit.set_phzvl      ( phzvl );      // Local zone word: (1*low) + (2*mid) + (4*low) - used in FW debugging
  //conv_hit.set_ph_hit     ( ph_hit );     // Intermediate quantity in phi calculation - used in FW debugging
  conv_hit.set_zone_hit   ( zone_hit );   // Phi value for building patterns (0.53333 deg precision)
  conv_hit.set_zone_code  ( zone_code );  // Full zone word: 1*(zone 0) + 2*(zone 1) + 4*(zone 2) + 8*(zone 3)

  conv_hit.set_fs_segment   ( fs_segment );    // Segment number used in primitive matching
  conv_hit.set_fs_zone_code ( fs_zone_code );  // Zone word used in primitive matching

  conv_hit.set_bt_station   ( bt_station );
  conv_hit.set_bt_segment   ( bt_segment );

  conv_hit.set_phi_loc  ( emtf::calc_phi_loc_deg(fph) );
  conv_hit.set_phi_glob ( emtf::calc_phi_glob_deg(conv_hit.Phi_loc(), conv_hit.Sector()) );
  conv_hit.set_theta    ( emtf::calc_theta_deg_from_int(th) );
  conv_hit.set_eta      ( emtf::calc_eta_from_theta_deg(conv_hit.Theta(), conv_hit.Endcap()) );
}
