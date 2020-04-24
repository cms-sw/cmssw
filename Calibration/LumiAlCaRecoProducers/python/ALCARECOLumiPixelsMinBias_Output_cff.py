import FWCore.ParameterSet.Config as cms

# AlCaReco for track based calibration using MinBias events
OutALCARECOLumiPixelsMinBias_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOLumiPixelsMinBias')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep *_siPixelClusters_*_*',
        'keep *_TriggerResults_*_HLT',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*', # not sure I'll need
        'keep *_TriggerResults_*_HLT',
        #'keep DcsStatuss_scalersRawToDigi_*_*', # fairly sure I don't need
        'keep *_offlinePrimaryVertices_*_*'
    )
)


import copy
OutALCARECOLumiPixelsMinBias=copy.deepcopy(OutALCARECOLumiPixelsMinBias_noDrop)
OutALCARECOLumiPixelsMinBias.outputCommands.insert(0,"drop *")
