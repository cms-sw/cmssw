import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using JetHT events
OutALCARECOTkAlJetHT_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlJetHT')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ALCARECOTkAlJetHT_*_*', 
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
        'keep *_offlinePrimaryVertices_*_*',
        'keep *_offlineBeamSpot_*_*')
)

OutALCARECOTkAlJetHT = OutALCARECOTkAlJetHT_noDrop.clone()
OutALCARECOTkAlJetHT.outputCommands.insert(0, "drop *")
