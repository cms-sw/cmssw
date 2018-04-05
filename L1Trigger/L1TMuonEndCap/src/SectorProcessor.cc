#include "L1Trigger/L1TMuonEndCap/interface/SectorProcessor.h"


SectorProcessor::SectorProcessor() {

}

SectorProcessor::~SectorProcessor() {

}

void SectorProcessor::configure(
    const GeometryTranslator* tp_geom,
    const ConditionHelper* cond,
    const SectorProcessorLUT* lut,
    PtAssignmentEngine** pt_assign_engine,
    int verbose, int endcap, int sector,
    int minBX, int maxBX, int bxWindow, int bxShiftCSC, int bxShiftRPC, int bxShiftGEM,
    const std::vector<int>& zoneBoundaries, int zoneOverlap,
    bool includeNeighbor, bool duplicateTheta, bool fixZonePhi, bool useNewZones, bool fixME11Edges,
    const std::vector<std::string>& pattDefinitions, const std::vector<std::string>& symPattDefinitions, bool useSymPatterns,
    int thetaWindow, int thetaWindowRPC, bool useSingleHits, bool bugSt2PhDiff, bool bugME11Dupes,
    int maxRoadsPerZone, int maxTracks, bool useSecondEarliest, bool bugSameSectorPt0,
    int ptLUTVersion, bool readPtLUTFile, bool fixMode15HighPt, bool bug9BitDPhi, bool bugMode7CLCT, bool bugNegPt, bool bugGMTPhi, bool promoteMode7
) {
  if (not(emtf::MIN_ENDCAP <= endcap && endcap <= emtf::MAX_ENDCAP))
    { edm::LogError("L1T") << "emtf::MIN_ENDCAP = " << emtf::MIN_ENDCAP 
			   << ", emtf::MAX_ENDCAP = " << emtf::MAX_ENDCAP
			   << ", endcap = " << endcap; return; }
  if (not(emtf::MIN_TRIGSECTOR <= sector && sector <= emtf::MAX_TRIGSECTOR))
    { edm::LogError("L1T") << "emtf::MIN_TRIGSECTOR = " << emtf::MIN_TRIGSECTOR 
			   << ", emtf::MAX_TRIGSECTOR = " << emtf::MAX_TRIGSECTOR
			   << ", endcap = " << sector; return; }
  if (not(tp_geom != nullptr))
    { edm::LogError("L1T") << "tp_geom = nullptr"; return; }
  if (not(cond != nullptr))
    { edm::LogError("L1T") << "cond = nullptr"; return; }
  if (not(lut != nullptr))
    { edm::LogError("L1T") << "lut = nullptr"; return; }
  if (not(pt_assign_engine != nullptr))
    { edm::LogError("L1T") << "pt_assign_engine = nullptr"; return; }

  tp_geom_          = tp_geom;
  cond_             = cond;
  lut_              = lut;
  pt_assign_engine_ = pt_assign_engine;

  verbose_  = verbose;
  endcap_   = endcap;
  sector_   = sector;

  minBX_       = minBX;
  maxBX_       = maxBX;
  bxWindow_    = bxWindow;
  bxShiftCSC_  = bxShiftCSC;
  bxShiftRPC_  = bxShiftRPC;
  bxShiftGEM_  = bxShiftGEM;

  zoneBoundaries_     = zoneBoundaries;
  zoneOverlap_        = zoneOverlap;
  includeNeighbor_    = includeNeighbor;
  duplicateTheta_     = duplicateTheta;
  fixZonePhi_         = fixZonePhi;
  useNewZones_        = useNewZones;
  fixME11Edges_       = fixME11Edges;

  pattDefinitions_    = pattDefinitions;
  symPattDefinitions_ = symPattDefinitions;
  useSymPatterns_     = useSymPatterns;

  thetaWindow_        = thetaWindow;
  thetaWindowRPC_     = thetaWindowRPC;
  useSingleHits_      = useSingleHits;
  bugSt2PhDiff_       = bugSt2PhDiff;
  bugME11Dupes_       = bugME11Dupes;

  maxRoadsPerZone_    = maxRoadsPerZone;
  maxTracks_          = maxTracks;
  useSecondEarliest_  = useSecondEarliest;
  bugSameSectorPt0_   = bugSameSectorPt0;

  ptLUTVersion_       = ptLUTVersion;
  readPtLUTFile_      = readPtLUTFile;
  fixMode15HighPt_    = fixMode15HighPt;
  bug9BitDPhi_        = bug9BitDPhi;
  bugMode7CLCT_       = bugMode7CLCT;
  bugNegPt_           = bugNegPt;
  bugGMTPhi_          = bugGMTPhi;
  promoteMode7_       = promoteMode7;
}

void SectorProcessor::set_pt_lut_version(unsigned pt_lut_version) {
  ptLUTVersion_ = pt_lut_version;
  // std::cout << "  * In endcap " << endcap_ << ", sector " << sector_ << ", set ptLUTVersion_ to " << ptLUTVersion_ << std::endl;
}

// Refer to docs/EMTF_FW_LUT_versions_2016_draft2.xlsx
void SectorProcessor::configure_by_fw_version(unsigned fw_version) {

  // std::cout << "Running configure_by_fw_version with version " << fw_version << std::endl;

  if (fw_version == 0 || fw_version == 123456)  // fw_version '123456' is from the fake conditions
    return;

  // For now, no switches later than FW version 47864 (end-of-year 2016)
  // Beggining in late 2016, "fw_version" in O2O populated with timestamp, rather than FW version
  // tm fw_time = gmtime(fw_version);  (See https://linux.die.net/man/3/gmtime)

  // Settings for 2017 (by default just use settings in simEmtfDigis_cfi.py)
  if (fw_version >= 50000) {

    // ___________________________________________________________________________
    // Versions in 2017 - no full documentation, can refer to https://twiki.cern.ch/twiki/bin/viewauth/CMS/L1KnownIssues
    
    // Before July 9th (runs < 298653), all mode 7 tracks (station 2-3-4) assigned quality 11
    // July 9th - 29th (runs 298653 - 300087), mode 7 tracks with |eta| > 1.6 in sector -6 assigned quality 12
    // After July 29th (runs >= 300088), mode 7 track promotion applied in all sectors
    // For some reason, the FW version in the database is 1496792995, at least for runs >= 298034 (July 4),
    //   which is towards the end of run 2017B (could not check earlier runs).  This corresponds to the date "June 6th",
    //   which is a month earlier than the first firmware update to apply this promotion.  So something's screwey.
    // Since July 18 is in the middle of the single-sector-fix period, would like to use a firmware version with
    //   roughly that date.  But this may require an intervention in the database. - AWB 04.08.17
    if (fw_version < 1496792995)
      promoteMode7_ = false;  // Assign station 2-3-4 tracks with |eta| > 1.6 SingleMu quality

    return;
  }

  // Settings for all of 2016 (following order in simEmtfDigis_cfi.py)
  else {
    minBX_      = -3;  // Minimum BX considered
    bxWindow_   =  3;  // Number of BX whose primitives can be included in the same track
    bxShiftCSC_ = -6;  // Shift applied to input CSC LCT primitives, to center at BX = 0

    zoneBoundaries_  = {0,41,49,87,127};  // Vertical boundaries of track-building zones, in integer theta
    zoneOverlap_     =  2;                // Overlap between zones
    includeNeighbor_ = true;              // Include primitives from neighbor chambers in track-building
    duplicateTheta_  = true;              // Use up to 4 theta/phi positions for two LCTs in the same chamber
    useNewZones_     = false;
    fixME11Edges_    = false;

    pattDefinitions_    = { "4,15:15,7:7,7:7,7:7",
			    "3,16:16,7:7,7:6,7:6",
			    "3,14:14,7:7,8:7,8:7",
			    "2,18:17,7:7,7:5,7:5",  // Should be 7:4 in ME3,4 (FW bug)
			    "2,13:12,7:7,10:7,10:7",
			    "1,22:19,7:7,7:0,7:0",
			    "1,11:8,7:7,14:7,14:7",
			    "0,30:23,7:7,7:0,7:0",
			    "0,7:0,7:7,14:7,14:7" };
    // Straightness, hits in ME1, hits in ME2, hits in ME3, hits in ME4
    symPattDefinitions_ = { "4,15:15:15:15,7:7:7:7,7:7:7:7,7:7:7:7",
			    "3,16:16:14:14,7:7:7:7,8:7:7:6,8:7:7:6",
			    "2,18:17:13:12,7:7:7:7,10:7:7:4,10:7:7:4",
			    "1,22:19:11:8,7:7:7:7,14:7:7:0,14:7:7:0",
			    "0,30:23:7:0,7:7:7:7,14:7:7:0,14:7:7:0" };

    thetaWindow_   = 4;      // Maximum dTheta between primitives in the same track
    useSingleHits_ = false;  // Build "tracks" from single LCTs in ME1/1

    maxRoadsPerZone_ = 3;  // Number of patterns that can be built per theta zone
    maxTracks_       = 3;  // Number of tracks that can be sent from each sector

    bugGMTPhi_ = true;
    promoteMode7_ = false;  // Assign station 2-3-4 tracks with |eta| > 1.6 SingleMu quality
  } // End default settings for 2016
			    

  // ___________________________________________________________________________
  // Versions in 2016 - refer to docs/EMTF_FW_LUT_versions_2016_draft2.xlsx

  // 1st_LCT_BX / 2nd_LCT_BX  (should also make unpacker configurable - AWB 21.07.17)
  // FW: Before: Earliest LCT used to assign BX, tracks only cancel within same BX
  //     After:  Second-earliest LCT used to assign BX, tracks cancel over 3 BX, improved LCT recovery
  useSecondEarliest_  = (fw_version < 46773) ? false : true;  // Changed Sept. 5

  // 8_BX_readout / 7_BX_readout
  // SW: DAQ readout changed from to [-3, +4] BX to [-3, +3] BX
  maxBX_              = (fw_version < 47109) ? +4 : +3;       // Changed Sept. 28

  // Asymm_patterns / Symm_patterns
  // FW: Changed from 9 asymmetric patterns to 5 symmetric patterns for track building
  useSymPatterns_     = (fw_version < 47214) ? false : true;  // Changed Oct. 6

  // HiPt_outlier
  // LUT: High-pT fix puts outlier LCTs in mode 15 tracks back in a straight line
  fixMode15HighPt_    = (fw_version < 46650) ? false : true;  // Changed July 25

  // Link_monitor (unpacker only)
  // FW: Added MPC link monitoring

  // ___________________________________________________________________________
  // Bugs

  // DAQ_ID (unpacker only; should make configurable - AWB 21.07.17)
  // FW: DAQ ME with output CSC ID range 0 - 8 instead of 1 - 9
  //     SP output ME2_ID, 3_ID, and 4_ID filled with 4, 5, or 6 when they should have been 7, 8, or 9.

  // ME_ID_FR
  // FW: Incorrect ME_ID fields in DAQ, wrong FR bits and some dPhi wrap-around in pT LUT address
  // - Unpacker only, or not worth emulating

  // DAQ_miss_LCT (unpacker only)
  // FW: LCTs only output if there was a track in the sector

  // Sector_pT_0
  // FW: Only highest-quality track in a sector assigned pT; others assigned pT = 0
  bugSameSectorPt0_   = (fw_version < 46650) ? true : false;  // Fixed July 22

  // Sector_bad_pT
  // FW: Tracks sometimes assigned pT of track in previous BX
  // - This is an ongoing (very rare) bug which occurs when 2 tracks try to access the same "bank" in the pT LUT
  //   It would be very difficult to emulate exactly, but the logic from Alex Madorsky is below
  // ## macro for detecting same bank address
  // ## bank and chip must match, and valid flags must be set
  // ## a and b are indexes 0,1,2
  // ## [X:Y] are bit portions from ptlut address words
  // `define sb(a,b) (ptlut_addr[a][29:26] == ptlut_addr[b][29:26] && ptlut_addr[a][5:2] == ptlut_addr[b][5:2] && ptlut_addr_val[a] && ptlut_addr_val[b])
  // ## This macro is used like this:
  // if (`sb(0,2) || `sb(1,2)) {disable PT readout for track 2}


  // DAQ_BX_3_LCT (unpacker only)
  // SW: LCTs in BX -3 only reported if there was a track in the sector
  // - not applicable

  // DAQ_BX_23_LCT (unpacker only)
  // SW: LCTs in BX -2 and -3 only reported if there was a track in the sector
  // - not applicable

  // pT_dPhi_bits
  // FW: dPhi wrap-around in modes 3, 5, 6, 9, 10, 12
  bug9BitDPhi_        = (fw_version < 47214) ? true : false;  // Fixed Oct. 6

  // Pattern_phi / ME1_neigh_phi
  // FW: Pattern phi slightly offset from true LCT phi; also ME3/4 pattern width off
  //     Pattern phi of neighbor hits in ME1 miscalculated
  fixZonePhi_         = (fw_version < 47214) ? false : true;  // Fixed Oct. 6

  // LCT_station_2
  // FW: Reduced LCT matching window in station 2, resulting in demoted tracks and inefficiency
  bugSt2PhDiff_       = (47109 <= fw_version && fw_version < 47249) ? true : false;  // Bug introduced Oct. 6, fixed Oct. 19

  // LCT_theta_dup
  // FW: LCTs matched to track may take theta value from other LCT in the same chamber
  bugME11Dupes_       = (fw_version < 47423) ? true : false;  // Fixed Nov. 1

  // LCT_7_10_neg_pT (E)
  // LUT: Written with incorrect values for mode 7 CLCT, mode 10 random offset, all modes negative (1/pT) set to 3 instead of 511
  bugMode7CLCT_       = (fw_version < 47864) ? true : false;  // Fixed sometime after Nov. 1
  bugNegPt_           = (fw_version < 47864) ? true : false;  // Fixed sometime after Nov. 1


}

void SectorProcessor::process(
    EventNumber_t ievent,
    const TriggerPrimitiveCollection& muon_primitives,
    EMTFHitCollection& out_hits,
    EMTFTrackCollection& out_tracks
) const {

  // List of converted hits, extended from previous BXs
  // deque (double-ended queue) is similar to a vector, but allows insertion or deletion of elements at both beginning and end
  std::deque<EMTFHitCollection> extended_conv_hits;

  // List of best track candidates, extended from previous BXs
  std::deque<EMTFTrackCollection> extended_best_track_cands;

  // Map of pattern detector --> lifetime, tracked across BXs
  std::map<pattern_ref_t, int> patt_lifetime_map;

  // ___________________________________________________________________________
  // Run each sector processor for every BX, taking into account the BX window

  int delayBX = bxWindow_ - 1;

  for (int bx = minBX_; bx <= maxBX_ + delayBX; ++bx) {
    if (verbose_ > 0) {  // debug
      std::cout << "Endcap: " << endcap_ << " Sector: " << sector_ << " Event: " << ievent << " BX: " << bx << std::endl;
    }

    process_single_bx(
        bx,
        muon_primitives,
        out_hits,
        out_tracks,
        extended_conv_hits,
        extended_best_track_cands,
        patt_lifetime_map
    );

    // Drop earliest BX outside of BX window
    if (bx >= minBX_ + delayBX) {
      extended_conv_hits.pop_front();

      int n = emtf::zone_array<int>().size();
      extended_best_track_cands.erase(extended_best_track_cands.end()-n, extended_best_track_cands.end());  // pop_back
    }
  }  // end loop over bx

  return;
}

void SectorProcessor::process_single_bx(
    int bx,
    const TriggerPrimitiveCollection& muon_primitives,
    EMTFHitCollection& out_hits,
    EMTFTrackCollection& out_tracks,
    std::deque<EMTFHitCollection>& extended_conv_hits,
    std::deque<EMTFTrackCollection>& extended_best_track_cands,
    std::map<pattern_ref_t, int>& patt_lifetime_map
) const {

  // ___________________________________________________________________________
  // Configure

  PrimitiveSelection prim_sel;
  prim_sel.configure(
      verbose_, endcap_, sector_, bx,
      bxShiftCSC_, bxShiftRPC_, bxShiftGEM_,
      includeNeighbor_, duplicateTheta_,
      bugME11Dupes_
  );

  PrimitiveConversion prim_conv;
  prim_conv.configure(
      tp_geom_, lut_,
      verbose_, endcap_, sector_, bx,
      bxShiftCSC_, bxShiftRPC_, bxShiftGEM_,
      zoneBoundaries_, zoneOverlap_,
      duplicateTheta_, fixZonePhi_, useNewZones_, fixME11Edges_,
      bugME11Dupes_
  );

  PatternRecognition patt_recog;
  patt_recog.configure(
      verbose_, endcap_, sector_, bx,
      bxWindow_,
      pattDefinitions_, symPattDefinitions_, useSymPatterns_,
      maxRoadsPerZone_, useSecondEarliest_
  );

  PrimitiveMatching prim_match;
  prim_match.configure(
      verbose_, endcap_, sector_, bx,
      fixZonePhi_, useNewZones_,
      bugSt2PhDiff_, bugME11Dupes_
  );

  AngleCalculation angle_calc;
  angle_calc.configure(
      verbose_, endcap_, sector_, bx,
      bxWindow_,
      thetaWindow_, thetaWindowRPC_,
      bugME11Dupes_
  );

  BestTrackSelection btrack_sel;
  btrack_sel.configure(
      verbose_, endcap_, sector_, bx,
      bxWindow_,
      maxRoadsPerZone_, maxTracks_, useSecondEarliest_,
      bugSameSectorPt0_
  );

  SingleHitTrack single_hit;
  single_hit.configure(
      verbose_, endcap_, sector_, bx,
      maxTracks_,
      useSingleHits_
  );

  PtAssignment pt_assign;
  pt_assign.configure(
      *pt_assign_engine_,
      verbose_, endcap_, sector_, bx,
      ptLUTVersion_, readPtLUTFile_, fixMode15HighPt_,
      bug9BitDPhi_, bugMode7CLCT_, bugNegPt_,
      bugGMTPhi_, promoteMode7_
  );

  std::map<int, TriggerPrimitiveCollection> selected_csc_map;
  std::map<int, TriggerPrimitiveCollection> selected_rpc_map;
  std::map<int, TriggerPrimitiveCollection> selected_gem_map;
  std::map<int, TriggerPrimitiveCollection> selected_prim_map;
  std::map<int, TriggerPrimitiveCollection> inclusive_selected_prim_map;

  EMTFHitCollection conv_hits;  // "converted" hits converted by primitive converter
  EMTFHitCollection inclusive_conv_hits;

  emtf::zone_array<EMTFRoadCollection> zone_roads;  // each zone has its road collection

  emtf::zone_array<EMTFTrackCollection> zone_tracks;  // each zone has its track collection

  EMTFTrackCollection best_tracks;  // "best" tracks selected from all the zones

  // ___________________________________________________________________________
  // Process

  // Select muon primitives that belong to this sector and this BX.
  // Put them into maps with an index that roughly corresponds to
  // each input link.
  // From src/PrimitiveSelection.cc
  prim_sel.process(CSCTag(), muon_primitives, selected_csc_map);
  prim_sel.process(RPCTag(), muon_primitives, selected_rpc_map);
  prim_sel.process(GEMTag(), muon_primitives, selected_gem_map);
  prim_sel.merge(selected_csc_map, selected_rpc_map, selected_gem_map, selected_prim_map);

  // Convert trigger primitives into "converted" hits
  // A converted hit consists of integer representations of phi, theta, and zones
  // From src/PrimitiveConversion.cc
  prim_conv.process(selected_prim_map, conv_hits);
  extended_conv_hits.push_back(conv_hits);

  {
    // Keep all the converted hits for the use of data-emulator comparisons.
    // They include the extra ones that are not used in track building and the subsequent steps.
    prim_sel.merge_no_truncate(selected_csc_map, selected_rpc_map, selected_gem_map, inclusive_selected_prim_map);
    prim_conv.process(inclusive_selected_prim_map, inclusive_conv_hits);
  }

  // Detect patterns in all zones, find 3 best roads in each zone
  // From src/PatternRecognition.cc
  patt_recog.process(extended_conv_hits, patt_lifetime_map, zone_roads);

  // Match the trigger primitives to the roads, create tracks
  // From src/PrimitiveMatching.cc
  prim_match.process(extended_conv_hits, zone_roads, zone_tracks);

  // Calculate deflection angles for each track and fill track variables
  // From src/AngleCalculation.cc
  angle_calc.process(zone_tracks);
  extended_best_track_cands.insert(extended_best_track_cands.begin(), zone_tracks.begin(), zone_tracks.end());  // push_front

  // Select 3 "best" tracks from all the zones
  // From src/BestTrackSelection.cc
  btrack_sel.process(extended_best_track_cands, best_tracks);

  // Insert single LCTs from station 1 as tracks
  // From src/SingleHitTracks.cc
  single_hit.process(conv_hits, best_tracks);

  // Construct pT address, assign pT, calculate other GMT quantities
  // From src/PtAssignment.cc
  pt_assign.process(best_tracks);

  // ___________________________________________________________________________
  // Output

  out_hits.insert(out_hits.end(), inclusive_conv_hits.begin(), inclusive_conv_hits.end());
  out_tracks.insert(out_tracks.end(), best_tracks.begin(), best_tracks.end());

  return;
}
