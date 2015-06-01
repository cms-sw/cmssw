import FWCore.ParameterSet.Config as cms

# AlCaReco output for track based muon alignment using cosmic ray tracks
OutALCARECOMuAlGlobalCosmics_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlGlobalCosmics')
    ),
    outputCommands = cms.untracked.vstring(
	'keep *_ALCARECOMuAlGlobalCosmics_*_*', # selected cosmic muons
	'keep *_ALCARECOMuAlCosmicsCTF_*_*',
	'keep *_ALCARECOMuAlCosmicsCosmicTF_*_*',
	'keep *_ALCARECOMuAlCosmicsRegional_*_*',
	'keep *_ALCARECOMuAlGeneralTracks_*_*', # selected general tracks
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
        'keep DcsStatuss_scalersRawToDigi_*_*')
)

import copy
OutALCARECOMuAlGlobalCosmics = copy.deepcopy(OutALCARECOMuAlGlobalCosmics_noDrop)
OutALCARECOMuAlGlobalCosmics.outputCommands.insert(0, "drop *")
