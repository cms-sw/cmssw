import FWCore.ParameterSet.Config as cms


# AlCaReco output for track based muon alignment using muons from collisions
OutALCARECOMuAlCalIsolatedMu_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlCalIsolatedMu')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ALCARECOMuAlCalIsolatedMu_*_*', # selected muons
        'keep *_ALCARECOMuAlCalIsolatedMuGeneralTracks_*_*', # selected general tracks
        'keep *_muonCSCDigis_*_*',
        'keep *_muonDTDigis_*_*',
        'keep *_muonRPCDigis_*_*',
        'keep *_dt1DRecHits_*_*', 
        'keep *_dt2DSegments_*_*', 
        'keep *_dt4DSegments_*_*', 
        'keep *_csc2DRecHits_*_*', 
        'keep *_cscSegments_*_*', 
	'keep *_gemRecHits_*_*', 
	'keep *_gemSegments_*_*',
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
OutALCARECOMuAlCalIsolatedMu = copy.deepcopy(OutALCARECOMuAlCalIsolatedMu_noDrop)
OutALCARECOMuAlCalIsolatedMu.outputCommands.insert(0, "drop *")
