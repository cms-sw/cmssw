import FWCore.ParameterSet.Config as cms

# EMTF emulator configuration
# Three options for CSCInput
#   * 'simCscTriggerPrimitiveDigis','MPCSORTED' : simulated trigger primitives (LCTs) from re-emulating CSC digis
#   * 'csctfDigis' : real trigger primitives as received by CSCTF (legacy trigger)
#   * 'emtfStage2Digis' : real trigger primitives as received by EMTF, unpacked in EventFilter/L1TRawToDigi/

# Check that proper switches are implemented in L1Trigger/Configuration/python/customiseReEmul.py - AWB 02.06.17

simEmtfDigisMC = cms.EDProducer("L1TMuonEndCapTrackProducer",
    # Verbosity level
    verbosity = cms.untracked.int32(0),

    # Input collections
    CSCInput = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED'),
    RPCInput = cms.InputTag('simMuonRPCDigis'),
    GEMInput = cms.InputTag('simMuonGEMPadDigis'),

    # Run with CSC, RPC, GEM
    CSCEnable = cms.bool(True),
    RPCEnable = cms.bool(True),
    GEMEnable = cms.bool(False),

    # BX
    MinBX    = cms.int32(-3),
    MaxBX    = cms.int32(+3),
    BXWindow = cms.int32(3),

    # CSC LCT BX offset correction
    CSCInputBXShift = cms.int32(-6),
    RPCInputBXShift = cms.int32(0),
    GEMInputBXShift = cms.int32(0),

    # Sector processor primitive-conversion parameters
    spPCParams16 = cms.PSet(
        ZoneBoundaries  = cms.vint32(0,41,49,87,127),  # 5 boundaries for 4 zones
        # ZoneBoundaries  = cms.vint32(0,36,54,96,127), # new proposed zone boundaries
        ZoneOverlap     = cms.int32(2),
        ZoneOverlapRPC  = cms.int32(8),
        IncludeNeighbor = cms.bool(True),
        DuplicateTheta  = cms.bool(True),
        FixZonePhi      = cms.bool(True),
        UseNewZones     = cms.bool(False),
        FixME11Edges    = cms.bool(True),
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
        UseSymmetricalPatterns = cms.bool(True),
    ),

    # Sector processor track-building parameters
    spTBParams16 = cms.PSet(
        ThetaWindow    = cms.int32(8),
        ThetaWindowRPC = cms.int32(8),
        UseSingleHits  = cms.bool(False),
        BugSt2PhDiff   = cms.bool(False),
        BugME11Dupes   = cms.bool(False),
    ),

    # Sector processor ghost-cancellation parameters
    spGCParams16 = cms.PSet(
        MaxRoadsPerZone   = cms.int32(3),
        MaxTracks         = cms.int32(3),
        UseSecondEarliest = cms.bool(True),
        BugSameSectorPt0  = cms.bool(False),
    ),

    # Sector processor pt-assignment parameters
    spPAParams16 = cms.PSet(
        PtLUTVersion    = cms.int32(7),  ## v5 for 2016, v6 for early 2017, v7 starting ~June 5 - AWB 02.06.17
        ReadPtLUTFile   = cms.bool(False),
        FixMode15HighPt = cms.bool(True),
        Bug9BitDPhi     = cms.bool(False),
        BugMode7CLCT    = cms.bool(False),
        BugNegPt        = cms.bool(False),
        BugGMTPhi       = cms.bool(False),
    ),

)

simEmtfDigisData = simEmtfDigisMC.clone(
    CSCInput = cms.InputTag('emtfStage2Digis'),
    RPCInput = cms.InputTag('muonRPCDigis'),
    GEMInput = cms.InputTag('muonGEMPadDigis'),
)

simEmtfDigis = simEmtfDigisMC.clone()
