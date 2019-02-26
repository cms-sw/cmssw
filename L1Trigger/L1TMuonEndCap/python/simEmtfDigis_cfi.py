import FWCore.ParameterSet.Config as cms

# EMTF emulator configuration

# Check that proper switches are implemented in L1Trigger/Configuration/python/customiseReEmul.py - AWB 02.06.17

###############################################################################################
###  IMPORTANT!!! Any changes to this file should be reflected in the 2016, 2017, and 2018  ###
###               configurations in configure_by_fw_version in src/SectorProcessor.cc       ###
###############################################################################################

simEmtfDigisMC = cms.EDProducer("L1TMuonEndCapTrackProducer",
    # Verbosity level
    verbosity = cms.untracked.int32(0),

    # Configure by firmware version, which may be different than the default parameters in this file
    FWConfig = cms.bool(True),

    # Input collections
    # Three options for CSCInput
    #   * 'simCscTriggerPrimitiveDigis','MPCSORTED' : simulated trigger primitives (LCTs) from re-emulating CSC digis
    #   * 'csctfDigis' : real trigger primitives as received by CSCTF (legacy trigger), available only in 2016 data
    #   * 'emtfStage2Digis' : real trigger primitives as received by EMTF, unpacked in EventFilter/L1TRawToDigi/
    CSCInput  = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED'),
    RPCInput  = cms.InputTag('simMuonRPCDigis'),
    CPPFInput = cms.InputTag('simCPPFDigis'),  ## Cannot use in MC workflow, does not exist yet.  CPPFEnable set to False - AWB 01.06.18
    GEMInput  = cms.InputTag('simMuonGEMPadDigis'),

    # Run with CSC, RPC, GEM
    CSCEnable = cms.bool(True),   # Use CSC LCTs from the MPCs in track-building
    RPCEnable = cms.bool(True),   # Use clustered RPC hits from CPPF in track-building
    CPPFEnable = cms.bool(False), # Use CPPF-emulated clustered RPC hits from CPPF as the RPC hits
    GEMEnable = cms.bool(False),  # Use hits from GEMs in track-building

    # Era (options: 'Run2_2016', 'Run2_2017', 'Run2_2018')
    Era = cms.string('Run2_2018'),

    # BX
    MinBX    = cms.int32(-3), # Minimum BX considered
    MaxBX    = cms.int32(+3), # Maximum BX considered
    BXWindow = cms.int32(2),  # Number of BX whose primitives can be included in the same track

    # CSC LCT BX offset correction
    CSCInputBXShift = cms.int32(-8), # Shift applied to input CSC LCT primitives, to center at BX = 0
    RPCInputBXShift = cms.int32(0),
    GEMInputBXShift = cms.int32(0),

    # Sector processor primitive-conversion parameters
    spPCParams16 = cms.PSet(
        PrimConvLUT     = cms.int32(2),    # v0, v1, and v2 LUTs used at different times, "-1" for local CPPF files (only works if FWConfig = False)
        ZoneBoundaries  = cms.vint32(0,41,49,87,127), # Vertical boundaries of track-building zones, in integer theta (5 for 4 zones)
        # ZoneBoundaries  = cms.vint32(0,36,54,96,127), # New proposed zone boundaries
        ZoneOverlap     = cms.int32(2),    # Overlap between zones
        IncludeNeighbor = cms.bool(True),  # Include primitives from neighbor chambers in track-building
        DuplicateTheta  = cms.bool(True),  # Use up to 4 theta/phi positions for two LCTs in the same chamber
        FixZonePhi      = cms.bool(True),  # Pattern phi slightly offset from true LCT phi; also ME3/4 pattern width off
        UseNewZones     = cms.bool(False), # Improve high-quality pattern finding near ring 1-2 gap in ME3/4
        FixME11Edges    = cms.bool(True),  # Improved small fraction of buggy LCT coordinate transformations
    ),

    # Sector processor pattern-recognition parameters
    spPRParams16 = cms.PSet(
        PatternDefinitions = cms.vstring(
            # straightness, hits in ME1, hits in ME2, hits in ME3, hits in ME4
            # ME1 vaues centered at 15, range from 0 - 30
            # ME2,3,4 values centered at 7, range from 0 - 14
            "4,15:15,7:7,7:7,7:7",
            "3,16:16,7:7,7:6,7:6",
            "3,14:14,7:7,8:7,8:7",
            "2,18:17,7:7,7:5,7:5",    # should be 7:4 in ME3,4 (FW bug)
            "2,13:12,7:7,10:7,10:7",
            "1,22:19,7:7,7:0,7:0",
            "1,11:8,7:7,14:7,14:7",
            "0,30:23,7:7,7:0,7:0",
            "0,7:0,7:7,14:7,14:7",
        ),
        SymPatternDefinitions = cms.vstring(
            # straightness, hits in ME1, hits in ME2, hits in ME3, hits in ME4
            "4,15:15:15:15,7:7:7:7,7:7:7:7,7:7:7:7",
            "3,16:16:14:14,7:7:7:7,8:7:7:6,8:7:7:6",
            "2,18:17:13:12,7:7:7:7,10:7:7:4,10:7:7:4",
            "1,22:19:11:8,7:7:7:7,14:7:7:0,14:7:7:0",
            "0,30:23:7:0,7:7:7:7,14:7:7:0,14:7:7:0",
        ),
        UseSymmetricalPatterns = cms.bool(True), # 5 symmetric patterns instead of 9 asymmetric for track building
    ),

    # Sector processor track-building parameters
    spTBParams16 = cms.PSet(
        ThetaWindow      = cms.int32(8),    # Maximum dTheta between primitives in the same track
        ThetaWindowZone0 = cms.int32(4),    # Maximum dTheta between primitives in the same track in Zone 0 (ring 1)
        UseSingleHits    = cms.bool(False), # Build "tracks" from single LCTs in ME1/1
        BugSt2PhDiff     = cms.bool(False), # Reduced LCT matching window in station 2, resulting in demoted tracks and inefficiency
        BugME11Dupes     = cms.bool(False), # LCTs matched to track may take theta value from other LCT in the same chamber
        BugAmbigThetaWin = cms.bool(False), # Can allow dThetas outside window when there are 2 LCTs in the same chamber
        TwoStationSameBX = cms.bool(True),  # Requires the hits in two-station tracks to have the same BX
    ),

    # Sector processor ghost-cancellation parameters
    spGCParams16 = cms.PSet(
        MaxRoadsPerZone   = cms.int32(3),    # Number of patterns that can be built per theta zone
        MaxTracks         = cms.int32(3),    # Number of tracks that can be sent from each sector
        UseSecondEarliest = cms.bool(True),  # Second-earliest LCT used to assign BX, tracks cancel over 3 BX, improved LCT recovery
        BugSameSectorPt0  = cms.bool(False), # Only highest-quality track in a sector assigned pT; others assigned pT = 0
    ),

    # Sector processor pt-assignment parameters
    spPAParams16 = cms.PSet(
        ReadPtLUTFile   = cms.bool(False),
        FixMode15HighPt = cms.bool(True),  # High-pT fix puts outlier LCTs in mode 15 tracks back in a straight line
        Bug9BitDPhi     = cms.bool(False), # dPhi wrap-around in modes 3, 5, 6, 9, 10, 12
        BugMode7CLCT    = cms.bool(False), # pT LUT written with incorrect values for mode 7 CLCT, mode 10 random offset
        BugNegPt        = cms.bool(False), # In all modes negative (1/pT) set to 3 instead of 511
        BugGMTPhi       = cms.bool(False), # Some drift in uGMT phi conversion, off by up to a few degrees
        PromoteMode7    = cms.bool(False), # Assign station 2-3-4 tracks with |eta| > 1.6 SingleMu quality
        ModeQualVer     = cms.int32(2),    # Version 2 contains modified mode-quality mapping for 2018
    ),

)

simEmtfDigisData = simEmtfDigisMC.clone(
    CSCInput  = cms.InputTag('emtfStage2Digis'),
    RPCInput  = cms.InputTag('muonRPCDigis'),
    CPPFInput = cms.InputTag('emtfStage2Digis'),
    GEMInput  = cms.InputTag('muonGEMPadDigis'),

    CPPFEnable = cms.bool(True), # Use CPPF-emulated clustered RPC hits from CPPF as the RPC hits

)

simEmtfDigis = simEmtfDigisMC.clone()


## Era: Run2_2016
#from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
#stage2L1Trigger.toModify(simEmtfDigis, RPCEnable = cms.bool(False), Era = cms.string('Run2_2016'))

## Era: Run2_2017
#from Configuration.Eras.Modifier_stage2L1Trigger_2017_cff import stage2L1Trigger_2017
#stage2L1Trigger_2017.toModify(simEmtfDigis, RPCEnable = cms.bool(True), Era = cms.string('Run2_2017'))
