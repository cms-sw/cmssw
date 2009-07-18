import FWCore.ParameterSet.Config as cms

# AlCaReco for track based calibration using MinBias events
OutALCARECOSiStripCalMinBias_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOSiStripCalMinBias')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep *_ALCARECOSiStripCalMinBias_*_*', 
        'keep *_siStripClusters_*_*', 
        'keep *_siPixelClusters_*_*', 
        'keep *_offlineBeamSpot_*_*')
)


import copy
OutALCARECOSiStripCalMinBias=copy.deepcopy(OutALCARECOSiStripCalMinBias_noDrop)
OutALCARECOSiStripCalMinBias.outputCommands.insert(0,"drop *")
