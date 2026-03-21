'''EventContent of Run-3 L1-Scouting data for the NANO(EDM)AOD data tiers'''
import FWCore.ParameterSet.Config as cms

L1SCOUTNanoAODEDMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        "drop *",

        # NanoAOD tables for products in L1-Scouting data
        "keep l1ScoutingRun3OrbitFlatTable_*_*_*",

        # vector of selected BXs in an orbit
        # (present only in some of the L1-Scouting data sets)
        "keep uints_*_*_*",
    )
)

L1SCOUTNANOAODEventContent = L1SCOUTNanoAODEDMEventContent.clone(
    compressionLevel = cms.untracked.int32(9),
    compressionAlgorithm = cms.untracked.string("LZMA"),
)
