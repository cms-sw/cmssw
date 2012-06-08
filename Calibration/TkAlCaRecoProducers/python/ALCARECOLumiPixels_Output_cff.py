import FWCore.ParameterSet.Config as cms

# AlCaReco for track based calibration using MinBias events
OutALCARECOLumiPixels_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOLumiPixels')
    ),
    outputCommands = cms.untracked.vstring( 
        "keep *_siPixelClustersForLumi_*_*",
        'keep *_TriggerResults_*_HLT')
)


import copy
OutALCARECOLumiPixels=copy.deepcopy(OutALCARECOLumiPixels_noDrop)
OutALCARECOLumiPixels.outputCommands.insert(0,"drop *")
