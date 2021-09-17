import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using ZMuMu events (including the tracks from the PV)
OutALCARECOTkAlDiMuonAndVertex_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlDiMuonAndVertex')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ALCARECOTkAlDiMuon_*_*', 
        'keep *_ALCARECOTkAlDiMuonVertexTracks_*_*',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
        'keep *_offlinePrimaryVertices_*_*')
)
OutALCARECOTkAlDiMuonAndVertex = OutALCARECOTkAlDiMuonAndVertex_noDrop.clone()
OutALCARECOTkAlDiMuonAndVertex.outputCommands.insert(0, "drop *")
