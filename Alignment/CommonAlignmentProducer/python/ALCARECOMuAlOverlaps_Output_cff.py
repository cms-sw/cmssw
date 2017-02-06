import FWCore.ParameterSet.Config as cms

# AlCaReco output for track based muon alignment using tracks in the CSC overlap regions
OutALCARECOMuAlOverlaps_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlOverlaps')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ALCARECOMuAlOverlaps_*_*', # selected muons through CSC overlap regions
        'keep *_ALCARECOMuAlOverlapsGeneralTracks_*_*', # selected general tracks
        'keep *_muonCSCDigis_*_*',
        'keep *_muonDTDigis_*_*',
        'keep *_muonRPCDigis_*_*',
        'keep *_dt1DRecHits_*_*',
        'keep *_dt2DSegments_*_*',
        'keep *_dt4DSegments_*_*',
        'keep *_csc2DRecHits_*_*',
        'keep *_cscSegments_*_*',
        'keep *_rpcRecHits_*_*',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep *_offlineBeamSpot_*_*',
        'keep *_offlinePrimaryVertices_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
    )
)

import copy
OutALCARECOMuAlOverlaps = copy.deepcopy(OutALCARECOMuAlOverlaps_noDrop)
OutALCARECOMuAlOverlaps.outputCommands.insert(0, "drop *")
