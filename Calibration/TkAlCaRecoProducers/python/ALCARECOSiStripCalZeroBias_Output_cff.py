import FWCore.ParameterSet.Config as cms

# AlCaReco for Bad Component Identification
OutALCARECOSiStripCalZeroBias_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOSiStripCalZeroBias')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep *_ALCARECOSiStripCalZeroBias_*_*',
        'keep *_calZeroBiasClusters_*_*',
        'keep *_MEtoEDMConverter_*_*')
)

import copy
OutALCARECOSiStripCalZeroBias=copy.deepcopy(OutALCARECOSiStripCalZeroBias_noDrop)
OutALCARECOSiStripCalZeroBias.insert(0,"drop *")
