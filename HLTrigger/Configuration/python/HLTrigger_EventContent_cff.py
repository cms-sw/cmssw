import FWCore.ParameterSet.Config as cms

# EventContent for HLT related products.

# This file exports the following five EventContent blocks:
#   HLTriggerRAW  HLTriggerRECO  HLTriggerAOD (without DEBUG products)
#   HLTDebugRAW   HLTDebugFEVT                (with    DEBUG products)
#
# as these are used in Configuration/EventContent
#
HLTriggerRAW = cms.PSet(
    outputCommands = cms.vstring('drop *_hlt*_*_*', 
        'keep *_hltL1GtObjectMap_*_*', 
        'keep FEDRawDataCollection_rawDataCollector_*_*', 
        'keep FEDRawDataCollection_source_*_*', 
        'keep edmTriggerResults_*_*_*', 
        'keep triggerTriggerEvent_*_*_*')
)

HLTriggerRECO = cms.PSet(
    outputCommands = cms.vstring('drop *_hlt*_*_*', 
        'keep *_hltL1GtObjectMap_*_*', 
        'keep edmTriggerResults_*_*_*', 
        'keep triggerTriggerEvent_*_*_*')
)

HLTriggerAOD = cms.PSet(
    outputCommands = cms.vstring('drop *_hlt*_*_*', 
        'keep *_hltL1GtObjectMap_*_*', 
        'keep edmTriggerResults_*_*_*', 
        'keep triggerTriggerEvent_*_*_*')
)

HLTDebugRAW = cms.PSet(
    outputCommands = cms.vstring('keep *')
)

HLTDebugFEVT = cms.PSet(
    outputCommands = cms.vstring('keep *')
)
