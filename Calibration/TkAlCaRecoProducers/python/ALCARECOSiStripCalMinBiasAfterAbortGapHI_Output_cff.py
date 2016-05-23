import FWCore.ParameterSet.Config as cms

# AlCaReco for track based calibration using MinBias events
OutALCARECOSiStripCalMinBiasAfterAbortGap_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOSiStripCalMinBiasAfterAbortGap')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep *_ALCARECOSiStripCalMinBiasAfterAbortGap_*_*', 
        'keep *_siStripClusters_*_*', 
        'keep *_siPixelClusters_*_*',
        'keep DetIdedmEDCollection_siStripDigis_*_*',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*')
)


import copy
OutALCARECOSiStripCalMinBiasAfterAbortGap=copy.deepcopy(OutALCARECOSiStripCalMinBiasAfterAbortGap_noDrop)
OutALCARECOSiStripCalMinBiasAfterAbortGap.outputCommands.insert(0,"drop *")
