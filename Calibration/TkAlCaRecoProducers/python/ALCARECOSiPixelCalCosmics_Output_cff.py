import FWCore.ParameterSet.Config as cms

OutALCARECOSiPixelCalCosmics_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOSiPixelCalCosmics')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ALCARECOSiPixelCalCosmics_*_*',
        'keep *_muons__*',
        'keep *_*riggerResults_*_HLT'
    )
)
import copy

OutALCARECOSiPixelCalCosmics=copy.deepcopy(OutALCARECOSiPixelCalCosmics_noDrop)
OutALCARECOSiPixelCalCosmics.outputCommands.insert(0, "drop *")
