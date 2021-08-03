import FWCore.ParameterSet.Config as cms

tmbPhase1 = cms.PSet(
    # block the ME1/a LCTs
    mpcBlockMe1a    = cms.uint32(0),
    # allow ALCT-only LCTs
    alctTrigEnable  = cms.uint32(0),
    # allow CLCT-only LCTs
    clctTrigEnable  = cms.uint32(0),
    # allow ALCT-CLCT correlated LCTs (default)
    matchTrigEnable = cms.uint32(1),
    # matching window for the trigger
    matchTrigWindowSize = cms.uint32(7),
    # array with the preferred delta BX values for the candidate CLCTs
    # perfectly in-time CLCTs are preferred, followed by
    # first-early, first-late, second-early, second late, etc.
    preferredBxMatch = cms.vint32(0, -1, 1, -2, 2, -3, 3),
    # readout window for the DAQ
    # LCTs found in the window [5, 6, 7, 8, 9, 10, 11] are good
    tmbL1aWindowSize = cms.uint32(7),
    # Debug
    verbosity = cms.int32(0),
    # Configure early_tbins instead of hardcoding it
    tmbEarlyTbins = cms.int32(4),
    # Flag for whether to readout only the earliest max two LCTs in a
    # L1A readout window, as there is only room just for two in the TMB header.
    # If false, all LCTs would be readout in L1A window.
    # originally, this planned to change this for Phase-2, but no longer
    # as of June 2021
    tmbReadoutEarliest2 = cms.bool(True),
    # For ALCT-centric matching in ME11, break after finding
    # the first BX with matching CLCT. Should always be set to True
    # when using the preferred BX windows
    matchEarliestClctOnly = cms.bool(True),
    # For ALCT-centric matching, whether to drop CLCTs that were matched
    # to ALCTs in this BX, and not use them in the following BX
    tmbDropUsedClcts = cms.bool(True),
    # True: allow construction of unphysical LCTs
    # in ME11 for which WG and HS do not intersect.
    # False: do not build such unphysical LCTs
    # It is recommended to keep this False, so that
    # the EMTF receives all information, physical or not
    ignoreAlctCrossClct = cms.bool(True),
    # bits for high-multiplicity triggers
    useHighMultiplicityBits = cms.bool(False),
)

# to be used by ME11 chambers with upgraded TMB and ALCT
tmbPhase2 = tmbPhase1.clone(
    # reduce ALCT-CLCT matching window size from 7 to 5
    matchTrigWindowSize = 5,
    # LCTs found in the window [6, 7, 8, 9, 10] are good
    tmbL1aWindowSize = 5,
    tmbDropUsedClcts = False,
)

tmbPhase2GEM = tmbPhase2.clone(
    # matching to GEM clusters in time
    maxDeltaBXALCTGEM = cms.uint32(1),
    maxDeltaBXCLCTGEM = cms.uint32(2),
    # True: use the half-strip coordinate to match
    # GEM with CSC TPs
    # False: use the eighth-strip coordinate + CCLUT slope
    # to match GEM with CSC TPs
    matchWithHS = cms.bool(True),
    # efficiency recovery switches
    dropLowQualityALCTsNoGEMs = cms.bool(False),
    dropLowQualityCLCTsNoGEMs = cms.bool(True),
    buildLCTfromALCTandGEM = cms.bool(True),
    buildLCTfromCLCTandGEM = cms.bool(True),
    # assign GEM-CSC bending angle. Works only for
    # Run-3 GEM-CSC TPs.
    assignGEMCSCBending = cms.bool(False),
    # whether to drop GEMs that were matched to ALCTs or CLCTs
    # in this BX, and not use them in the following BX
    tmbDropUsedGems = cms.bool(False),
    # For ALCT-centric matching to GEMs, break after finding
    # the first BX with matching GEM coincidence clusters
    matchEarliestGemsOnly = cms.bool(True),
    # When running the GEM-CSC matching, whether to mitigate
    # the slope of CLCTs with high, meaning inconsistent,
    # values of cosi (consistency of slope indicator)
    # to optimize GEM-CSC slope correction
    mitigateSlopeByCosi = cms.bool(True),
)

# to be used by ME11 chambers with GEM-CSC ILT
tmbPhase2GE11 = tmbPhase2GEM.clone(
    # matching to GEM clusters with half-strips
    maxDeltaHsEven = cms.uint32(7),
    maxDeltaHsOdd = cms.uint32(16),
    maxDeltaHsEvenME1a = cms.uint32(5),
    maxDeltaHsOddME1a = cms.uint32(12),
    # efficiency recovery switches
    dropLowQualityCLCTsNoGEMs_ME1a = cms.bool(True),
    buildLCTfromALCTandGEM_ME1a = cms.bool(True),
    buildLCTfromCLCTandGEM_ME1a = cms.bool(True),
)

# to be used by ME21 chambers with GEM-CSC ILT
tmbPhase2GE21 = tmbPhase2GEM.clone(
    # matching to GEM clusters with half-strips
    maxDeltaHsEven = cms.uint32(3),
    maxDeltaHsOdd = cms.uint32(5),
    dropLowQualityALCTsNoGEMs = True,
)

# LUTs to map wiregroup onto min and max half-strip number that it crosses in ME1/1
wgCrossHsME11Params = cms.PSet(
    wgCrossHsME1aFiles = cms.vstring(
        "L1Trigger/CSCTriggerPrimitives/data/ME11/CSCLUT_wg_min_hs_ME1a.txt",
        "L1Trigger/CSCTriggerPrimitives/data/ME11/CSCLUT_wg_max_hs_ME1a.txt",
    ),
    wgCrossHsME1aGangedFiles = cms.vstring(
        "L1Trigger/CSCTriggerPrimitives/data/ME11/CSCLUT_wg_min_hs_ME1a_ganged.txt",
        "L1Trigger/CSCTriggerPrimitives/data/ME11/CSCLUT_wg_max_hs_ME1a_ganged.txt",
    ),
    wgCrossHsME1bFiles = cms.vstring(
        "L1Trigger/CSCTriggerPrimitives/data/ME11/CSCLUT_wg_min_hs_ME1b.txt",
        "L1Trigger/CSCTriggerPrimitives/data/ME11/CSCLUT_wg_max_hs_ME1b.txt",
    )
)

# LUTs with correspondence between ALCT-CLCT combination
# code and the resulting best/second lct1
lctCodeParams = cms.PSet(
    lctCodeFiles = cms.vstring(
        "L1Trigger/CSCTriggerPrimitives/data/LCTCode/CSCLUT_code_to_bestLCT.txt",
        "L1Trigger/CSCTriggerPrimitives/data/LCTCode/CSCLUT_code_to_secondLCT.txt",
    )
)

tmbPSets = cms.PSet(
    tmbPhase1 = tmbPhase1.clone(),
    tmbPhase2 = tmbPhase2.clone(),
    tmbPhase2GE11 = tmbPhase2GE11.clone(),
    tmbPhase2GE21 = tmbPhase2GE21.clone(),
    wgCrossHsME11Params = wgCrossHsME11Params.clone(),
    lctCodeParams = lctCodeParams.clone()
)
