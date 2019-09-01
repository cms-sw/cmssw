import FWCore.ParameterSet.Config as cms

# AlCaReco for Bad Component Identification
OutALCARECOSiPixelCalZeroBias_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOSiPixelCalZeroBias')
    ),
    outputCommands=cms.untracked.vstring(   
        'keep *_ALCARECOSiPixelCalZeroBias_*_*',
        'keep *_siPixelStatusProducer_*_*')
)

import copy
OutALCARECOSiPixelCalZeroBias=copy.deepcopy(OutALCARECOSiPixelCalZeroBias_noDrop)
OutALCARECOSiPixelCalZeroBias.outputCommands.insert(0,"drop *")

