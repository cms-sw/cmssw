#include "L1Trigger/L1TMuonEndCap/interface/VersionControl.h"

VersionControl::VersionControl(const edm::ParameterSet& iConfig) : config_(iConfig) {
  // All the configurables from python/simEmtfDigis_cfi.py must be visible to this class, except InputTags.
  verbose_ = iConfig.getUntrackedParameter<int>("verbosity");
  useO2O_ = iConfig.getParameter<bool>("FWConfig");
  era_ = iConfig.getParameter<std::string>("Era");
  // Run 3 CCLUT
  useRun3CCLUT_ = iConfig.getParameter<bool>("UseRun3CCLUT");

  useDT_ = iConfig.getParameter<bool>("DTEnable");
  useCSC_ = iConfig.getParameter<bool>("CSCEnable");
  useRPC_ = iConfig.getParameter<bool>("RPCEnable");
  useIRPC_ = iConfig.getParameter<bool>("IRPCEnable");
  useCPPF_ = iConfig.getParameter<bool>("CPPFEnable");
  useGEM_ = iConfig.getParameter<bool>("GEMEnable");
  useME0_ = iConfig.getParameter<bool>("ME0Enable");

  minBX_ = iConfig.getParameter<int>("MinBX");
  maxBX_ = iConfig.getParameter<int>("MaxBX");
  bxWindow_ = iConfig.getParameter<int>("BXWindow");
  bxShiftCSC_ = iConfig.getParameter<int>("CSCInputBXShift");
  bxShiftRPC_ = iConfig.getParameter<int>("RPCInputBXShift");
  bxShiftGEM_ = iConfig.getParameter<int>("GEMInputBXShift");
  bxShiftME0_ = iConfig.getParameter<int>("ME0InputBXShift");

  auto spPCParams16 = iConfig.getParameter<edm::ParameterSet>("spPCParams16");
  zoneBoundaries_ = spPCParams16.getParameter<std::vector<int> >("ZoneBoundaries");
  zoneOverlap_ = spPCParams16.getParameter<int>("ZoneOverlap");
  includeNeighbor_ = spPCParams16.getParameter<bool>("IncludeNeighbor");
  duplicateTheta_ = spPCParams16.getParameter<bool>("DuplicateTheta");
  fixZonePhi_ = spPCParams16.getParameter<bool>("FixZonePhi");
  useNewZones_ = spPCParams16.getParameter<bool>("UseNewZones");
  fixME11Edges_ = spPCParams16.getParameter<bool>("FixME11Edges");

  auto spPRParams16 = iConfig.getParameter<edm::ParameterSet>("spPRParams16");
  pattDefinitions_ = spPRParams16.getParameter<std::vector<std::string> >("PatternDefinitions");
  symPattDefinitions_ = spPRParams16.getParameter<std::vector<std::string> >("SymPatternDefinitions");
  useSymPatterns_ = spPRParams16.getParameter<bool>("UseSymmetricalPatterns");

  auto spTBParams16 = iConfig.getParameter<edm::ParameterSet>("spTBParams16");
  thetaWindow_ = spTBParams16.getParameter<int>("ThetaWindow");
  thetaWindowZone0_ = spTBParams16.getParameter<int>("ThetaWindowZone0");
  useSingleHits_ = spTBParams16.getParameter<bool>("UseSingleHits");
  bugSt2PhDiff_ = spTBParams16.getParameter<bool>("BugSt2PhDiff");
  bugME11Dupes_ = spTBParams16.getParameter<bool>("BugME11Dupes");
  bugAmbigThetaWin_ = spTBParams16.getParameter<bool>("BugAmbigThetaWin");
  twoStationSameBX_ = spTBParams16.getParameter<bool>("TwoStationSameBX");

  auto spGCParams16 = iConfig.getParameter<edm::ParameterSet>("spGCParams16");
  maxRoadsPerZone_ = spGCParams16.getParameter<int>("MaxRoadsPerZone");
  maxTracks_ = spGCParams16.getParameter<int>("MaxTracks");
  useSecondEarliest_ = spGCParams16.getParameter<bool>("UseSecondEarliest");
  bugSameSectorPt0_ = spGCParams16.getParameter<bool>("BugSameSectorPt0");

  auto spPAParams16 = iConfig.getParameter<edm::ParameterSet>("spPAParams16");
  readPtLUTFile_ = spPAParams16.getParameter<bool>("ReadPtLUTFile");
  fixMode15HighPt_ = spPAParams16.getParameter<bool>("FixMode15HighPt");
  bug9BitDPhi_ = spPAParams16.getParameter<bool>("Bug9BitDPhi");
  bugMode7CLCT_ = spPAParams16.getParameter<bool>("BugMode7CLCT");
  bugNegPt_ = spPAParams16.getParameter<bool>("BugNegPt");
  bugGMTPhi_ = spPAParams16.getParameter<bool>("BugGMTPhi");
  promoteMode7_ = spPAParams16.getParameter<bool>("PromoteMode7");
  modeQualVer_ = spPAParams16.getParameter<int>("ModeQualVer");
  pbFileName_ = spPAParams16.getParameter<std::string>("ProtobufFileName");
}

VersionControl::~VersionControl() {}

// Refer to docs/EMTF_FW_LUT_versions_2016_draft2.xlsx
void VersionControl::configure_by_fw_version(unsigned fw_version) {
  if (fw_version == 0 || fw_version == 123456)  // fw_version '123456' is from the fake conditions
    return;

  // For now, no switches later than FW version 47864 (end-of-year 2016)
  // Beggining in late 2016, "fw_version" in O2O populated with timestamp, rather than FW version
  // tm fw_time = gmtime(fw_version);  (See https://linux.die.net/man/3/gmtime, https://www.epochconverter.com)

  /////////////////////////////////////////////////////////////////////////////////
  ///  Settings for 2018 (by default just use settings in simEmtfDigis_cfi.py)  ///
  /////////////////////////////////////////////////////////////////////////////////
  if (fw_version >= 1514764800) {  // January 1, 2018

    // Settings for all of 2018 (following order in simEmtfDigis_cfi.py)
    // BXWindow(2) and BugAmbigThetaWin(F) deployed sometime before stable beams on March 20, not quite sure when - AWB 26.04.18
    // TwoStationSameBX(T), ThetaWindowZone0(4), and ModeQualVer(2) to be deployed sometime between May 17 and May 31 - AWB 14.05.18

    // Global parameters
    // Defaults : CSCEnable(T), RPCEnable(T), GEMEnable(F), Era("Run2_2018"), MinBX(-3), MaxBX(+3), BXWindow(2)
    // --------------------------------------------------------------------------------------------------------
    era_ = "Run2_2018";  // Era for CMSSW customization
    bxWindow_ = 2;       // Number of BX whose primitives can be included in the same track

    // spTBParams16 : Sector processor track-building parameters
    // Defaults : PrimConvLUT(1), ZoneBoundaries(0,41,49,87,127), ZoneOverlap(2), IncludeNeighbor(T),
    //            DuplicateThteta(T), FixZonePhi(T), UseNewZones(F), FixME11Edges(T)
    // ------------------------------------------------------------------------------

    // spPRParams16 : Sector processor pattern-recognition parameters
    // Defaults : PatternDefinitions(...), SymPatternDefinitions(...), UseSymmetricalPatterns(T)
    // -----------------------------------------------------------------------------------------

    // spTBParams16 : Sector processor track-building parameters
    // Defaults : ThetaWindow(8), ThetaWindowZone0(4), UseSingleHits(F), BugSt2PhDiff(F),
    //            BugME11Dupes(F), BugAmbigThetaWin(F), TwoStationSameBX(T)
    // ----------------------------------------------------------------------------------
    thetaWindow_ = 8;           // Maximum dTheta between primitives in the same track
    thetaWindowZone0_ = 4;      // Maximum dTheta between primitives in the same track in Zone 0 (ring 1)
    bugAmbigThetaWin_ = false;  // Can allow dThetas outside window when there are 2 LCTs in the same chamber
    twoStationSameBX_ = true;   // Requires the hits in two-station tracks to have the same BX

    // spGCParams16 : Sector processor ghost-cancellation parameters
    // Defaults : MaxRoadsPerZone(3), MaxTracks(3), UseSecondEarliest(T), BugSameSectorPt0(F)
    // --------------------------------------------------------------------------------------

    // spPAParams16 : Sector processor pt-assignment parameters
    // Defaults : ReadPtLUTFile(F), FixMode15HighPt(T), Bug9BitDPhi(F), BugMode7CLCT(F),
    //            BugNegPt(F), BugGMTPhi(F), PromoteMode7(F), ModeQualVer(2)
    // ---------------------------------------------------------------------------------
    modeQualVer_ = 2;       // Version 2 contains modified mode-quality mapping for 2018
    promoteMode7_ = false;  // Assign station 2-3-4 tracks with |eta| > 1.6 SingleMu quality

    // ___________________________________________________________________________
    // Versions in 2018 - no external documentation
    // As of the beginning of 2018 EMTF O2O was broken, not updating the database with online conditions
    // Firmware version reported for runs before 318841 (June 28) is 1504018578 (Aug. 29, 2017) even though
    //   updates occured in February and March of 2018.  May need to re-write history in the database. - AWB 30.06.18

    if (fw_version < 1529520380) {  // June 20, 2018
      thetaWindowZone0_ = 8;        // Maximum dTheta between primitives in the same track in Zone 0 (ring 1)
      twoStationSameBX_ = false;    // Requires the hits in two-station tracks to have the same BX
      modeQualVer_ = 1;             // Version 2 contains modified mode-quality mapping for 2018
      promoteMode7_ = true;         // Assign station 2-3-4 tracks with |eta| > 1.6 SingleMu quality
    }

    return;
  }

  /////////////////////////////////////////////////////////////////////////////////
  ///  Settings for 2017 (by default just use settings in simEmtfDigis_cfi.py)  ///
  /////////////////////////////////////////////////////////////////////////////////
  else if (fw_version >= 50000) {
    // Settings for all of 2017 (following order in simEmtfDigis_cfi.py)

    // Global parameters
    // Defaults : CSCEnable(T), RPCEnable(T), GEMEnable(F), Era("Run2_2018"), MinBX(-3), MaxBX(+3), BXWindow(2)
    // --------------------------------------------------------------------------------------------------------
    era_ = "Run2_2017";  // Era for CMSSW customization
    bxWindow_ = 3;       // Number of BX whose primitives can be included in the same track

    // spTBParams16 : Sector processor track-building parameters
    // Defaults : PrimConvLUT(1), ZoneBoundaries(0,41,49,87,127), ZoneOverlap(2), IncludeNeighbor(T),
    //            DuplicateThteta(T), FixZonePhi(T), UseNewZones(F), FixME11Edges(T)
    // ------------------------------------------------------------------------------

    // spPRParams16 : Sector processor pattern-recognition parameters
    // Defaults : PatternDefinitions(...), SymPatternDefinitions(...), UseSymmetricalPatterns(T)
    // -----------------------------------------------------------------------------------------

    // spTBParams16 : Sector processor track-building parameters
    // Defaults : ThetaWindow(8), ThetaWindowZone0(4), UseSingleHits(F), BugSt2PhDiff(F),
    //            BugME11Dupes(F), BugAmbigThetaWin(F), TwoStationSameBX(T)
    // ----------------------------------------------------------------------------------
    thetaWindow_ = 8;           // Maximum dTheta between primitives in the same track
    thetaWindowZone0_ = 8;      // Maximum dTheta between primitives in the same track in Zone 0 (ring 1)
    bugAmbigThetaWin_ = true;   // Can allow dThetas outside window when there are 2 LCTs in the same chamber
    twoStationSameBX_ = false;  // Requires the hits in two-station tracks to have the same BX

    // spGCParams16 : Sector processor ghost-cancellation parameters
    // Defaults : MaxRoadsPerZone(3), MaxTracks(3), UseSecondEarliest(T), BugSameSectorPt0(F)
    // --------------------------------------------------------------------------------------

    // spPAParams16 : Sector processor pt-assignment parameters
    // Defaults : ReadPtLUTFile(F), FixMode15HighPt(T), Bug9BitDPhi(F), BugMode7CLCT(F),
    //            BugNegPt(F), BugGMTPhi(F), PromoteMode7(F)
    // ---------------------------------------------------------------------------------
    modeQualVer_ = 1;  // Version 2 contains modified mode-quality mapping for 2018

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
    // Last firmware version in 2017 was 1504018578 (Aug. 29, 2017).
    if (fw_version >= 1496792995)
      promoteMode7_ = true;  // Assign station 2-3-4 tracks with |eta| > 1.6 SingleMu quality

    return;
  }

  ///////////////////////////////////////////////////////////////////////////
  ///  Settings for all of 2016 (following order in simEmtfDigis_cfi.py)  ///
  ///////////////////////////////////////////////////////////////////////////
  else {
    // Global parameters
    // Defaults : CSCEnable(T), RPCEnable(T), GEMEnable(F), Era("Run2_2018"), MinBX(-3), MaxBX(+3), BXWindow(2)
    // --------------------------------------------------------------------------------------------------------
    useRPC_ = false;     // Use clustered RPC hits from CPPF in track-building
    era_ = "Run2_2016";  // Era for CMSSW customization
    // maxBX_                 // Depends on FW version, see below
    bxWindow_ = 3;  // Number of BX whose primitives can be included in the same track

    // spTBParams16 : Sector processor track-building parameters
    // Defaults : PrimConvLUT(1), ZoneBoundaries(0,41,49,87,127), ZoneOverlap(2), IncludeNeighbor(T),
    //            DuplicateThteta(T), FixZonePhi(T), UseNewZones(F), FixME11Edges(T)
    // ------------------------------------------------------------------------------
    // primConvLUT_         // Should be 0 for 2016, set using get_pc_lut_version() from ConditionsHelper.cc
    // fixZonePhi_          // Depends on FW version, see below
    fixME11Edges_ = false;  // Improved small fraction of buggy LCT coordinate transformations

    // spPRParams16 : Sector processor pattern-recognition parameters
    // Defaults : PatternDefinitions(...), SymPatternDefinitions(...), UseSymmetricalPatterns(T)
    // -----------------------------------------------------------------------------------------
    // useSymPatterns_  // Depends on FW version, see below

    // spTBParams16 : Sector processor track-building parameters
    // Defaults : ThetaWindow(8), ThetaWindowZone0(4), UseSingleHits(F), BugSt2PhDiff(F),
    //            BugME11Dupes(F), BugAmbigThetaWin(F), TwoStationSameBX(T)
    // ----------------------------------------------------------------------------------
    thetaWindow_ = 4;       // Maximum dTheta between primitives in the same track
    thetaWindowZone0_ = 4;  // Maximum dTheta between primitives in the same track in Zone 0 (ring 1)
    // bugSt2PhDiff_           // Depends on FW version, see below
    // bugME11Dupes_           // Depends on FW version, see below
    bugAmbigThetaWin_ = true;   // Can allow dThetas outside window when there are 2 LCTs in the same chamber
    twoStationSameBX_ = false;  // Requires the hits in two-station tracks to have the same BX

    // spGCParams16 : Sector processor ghost-cancellation parameters
    // Defaults : MaxRoadsPerZone(3), MaxTracks(3), UseSecondEarliest(T), BugSameSectorPt0(F)
    // --------------------------------------------------------------------------------------
    // useSecondEarliest_  // Depends on FW version, see below
    // bugSameSectorPt0_   // Depends on FW version, see below

    // spPAParams16 : Sector processor pt-assignment parameters
    // Defaults : ReadPtLUTFile(F), FixMode15HighPt(T), Bug9BitDPhi(F), BugMode7CLCT(F),
    //            BugNegPt(F), BugGMTPhi(F), PromoteMode7(F)
    // ---------------------------------------------------------------------------------
    // fixMode15HighPt_   // Depends on FW version, see below
    // bug9BitDPhi_       // Depends on FW version, see below
    // bugMode7CLCT_      // Depends on FW version, see below
    // bugNegPt_          // Depends on FW version, see below
    bugGMTPhi_ = true;  // Some drift in uGMT phi conversion, off by up to a few degrees
    modeQualVer_ = 1;   // Version 2 contains modified mode-quality mapping for 2018

  }  // End default settings for 2016

  // ___________________________________________________________________________
  // Versions in 2016 - refer to docs/EMTF_FW_LUT_versions_2016_draft2.xlsx

  // 1st_LCT_BX / 2nd_LCT_BX  (should also make unpacker configurable - AWB 21.07.17)
  // FW: Before: Earliest LCT used to assign BX, tracks only cancel within same BX
  //     After:  Second-earliest LCT used to assign BX, tracks cancel over 3 BX, improved LCT recovery
  useSecondEarliest_ = (fw_version < 46773) ? false : true;  // Changed Sept. 5

  // 8_BX_readout / 7_BX_readout
  // SW: DAQ readout changed from to [-3, +4] BX to [-3, +3] BX
  maxBX_ = (fw_version < 47109) ? +4 : +3;  // Changed Sept. 28

  // Asymm_patterns / Symm_patterns
  // FW: Changed from 9 asymmetric patterns to 5 symmetric patterns for track building
  useSymPatterns_ = (fw_version < 47214) ? false : true;  // Changed Oct. 6

  // HiPt_outlier
  // LUT: High-pT fix puts outlier LCTs in mode 15 tracks back in a straight line
  fixMode15HighPt_ = (fw_version < 46650) ? false : true;  // Changed July 25

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
  bugSameSectorPt0_ = (fw_version < 46650) ? true : false;  // Fixed July 22

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
  bug9BitDPhi_ = (fw_version < 47214) ? true : false;  // Fixed Oct. 6

  // Pattern_phi / ME1_neigh_phi
  // FW: Pattern phi slightly offset from true LCT phi; also ME3/4 pattern width off
  //     Pattern phi of neighbor hits in ME1 miscalculated
  fixZonePhi_ = (fw_version < 47214) ? false : true;  // Fixed Oct. 6

  // LCT_station_2
  // FW: Reduced LCT matching window in station 2, resulting in demoted tracks and inefficiency
  bugSt2PhDiff_ = (47109 <= fw_version && fw_version < 47249) ? true : false;  // Bug introduced Oct. 6, fixed Oct. 19

  // LCT_theta_dup
  // FW: LCTs matched to track may take theta value from other LCT in the same chamber
  bugME11Dupes_ = (fw_version < 47423) ? true : false;  // Fixed Nov. 1

  // LCT_7_10_neg_pT (E)
  // LUT: Written with incorrect values for mode 7 CLCT, mode 10 random offset, all modes negative (1/pT) set to 3 instead of 511
  bugMode7CLCT_ = (fw_version < 47864) ? true : false;  // Fixed sometime after Nov. 1
  bugNegPt_ = (fw_version < 47864) ? true : false;      // Fixed sometime after Nov. 1
}
