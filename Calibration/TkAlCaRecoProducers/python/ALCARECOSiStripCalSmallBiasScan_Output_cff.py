import FWCore.ParameterSet.Config as cms

# AlCaReco for track based calibration using MinBias events
OutALCARECOSiStripCalSmallBiasScan_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOSiStripCalSmallBiasScan')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep *_ALCARECOSiStripCalSmallBiasScan_*_*', 
        'keep *_siStripClusters_*_*', 
        'keep *_siPixelClusters_*_*',
        'keep DetIdedmEDCollection_siStripDigis_*_*',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep LumiScalerss_scalersRawToDigi_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
        'keep *_TriggerResults_*_*')
)


import copy
OutALCARECOSiStripCalSmallBiasScan=copy.deepcopy(OutALCARECOSiStripCalSmallBiasScan_noDrop)
OutALCARECOSiStripCalSmallBiasScan.outputCommands.insert(0,"drop *")
