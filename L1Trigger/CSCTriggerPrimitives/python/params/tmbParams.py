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
    tmbDropUsedClcts = cms.bool(False),
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
    # ALCT-CLCT stays at 7 for the moment
    matchTrigWindowSize = 7,
    # LCTs found in the window [5, 6, 7, 8, 9, 10, 11] are good
    tmbL1aWindowSize = 7,
    tmbDropUsedClcts = False,
)

tmbPhase2GEM = tmbPhase2.clone(
    # matching to GEM clusters in time
    windowBXALCTGEM = cms.uint32(3),
    windowBXCLCTGEM = cms.uint32(7),
    # Set to True for matching CLCT and GEM clusters by "propagating"
    # CLCT slope to GEM. Otherwise, just matching keystrip positions
    matchCLCTpropagation = cms.bool(True),
    # efficiency recovery switches
    dropLowQualityALCTs = cms.bool(True),
    dropLowQualityCLCTs = cms.bool(True),
    buildLCTfromALCTandGEM = cms.bool(True),
    buildLCTfromCLCTandGEM = cms.bool(False),
    # assign GEM-CSC bending angle. Works only for
    # Run-3 GEM-CSC TPs.
    assignGEMCSCBending = cms.bool(True),
    # When running the GEM-CSC matching, whether to mitigate
    # the slope of CLCTs with high, meaning inconsistent,
    # values of cosi (consistency of slope indicator)
    # to optimize GEM-CSC slope correction
    mitigateSlopeByCosi = cms.bool(False),
    # Preferred order of bunchcrossing difference between CSC minus GEM BX for matching
    BunchCrossingCSCminGEMwindow = cms.vint32(0, -1, 1, -2, 2, -3, 3)
)

# to be used by ME11 chambers with GEM-CSC ILT
tmbPhase2GE11 = tmbPhase2GEM.clone(
    # match ME1a with GEM (it affects CLCT-GEM match)
    enableMatchGEMandME1a = cms.bool(True),
    # match ME1b with GEM (it affects CLCT-GEM match)
    enableMatchGEMandME1b = cms.bool(True),
    # matching windows for ALCT-GEM clusters in wiregroups
    maxDeltaWG = cms.uint32(7),
    # matching windows for CLCT-GEM clusters in half strip units
    maxDeltaHsEven = cms.uint32(5),
    maxDeltaHsOdd = cms.uint32(10),
    # efficiency recovery switches
    dropLowQualityCLCTs_ME1a = cms.bool(True),
    # delay applied in OTMB to GEM trigger primitives
    delayGEMinOTMB = cms.uint32(0)
)

# to be used by ME21 chambers with GEM-CSC ILT
tmbPhase2GE21 = tmbPhase2GEM.clone(
    # match ME1a with GEM (it affects CLCT-GEM match)
    enableMatchGEMandME1a = cms.bool(True),
    # match ME1b with GEM (it affects CLCT-GEM match)
    enableMatchGEMandME1b = cms.bool(True),
    # matching windows for ALCT-GEM clusters in wiregroups
    maxDeltaWG = cms.uint32(7),
    # matching windows for CLCT-GEM clusters in half strip units
    maxDeltaHsEven = cms.uint32(5),
    maxDeltaHsOdd = cms.uint32(10),
    # efficiency recovery switches
    dropLowQualityCLCTs_ME1a = cms.bool(True),
    # delay applied in OTMB to GEM trigger primitives
    delayGEMinOTMB = cms.uint32(0)
)

tmbPSets = cms.PSet(
    tmbPhase1 = tmbPhase1.clone(),
    tmbPhase2 = tmbPhase2.clone(),
    tmbPhase2GE11 = tmbPhase2GE11.clone(),
    tmbPhase2GE21 = tmbPhase2GE21.clone()
)
