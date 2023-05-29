import FWCore.ParameterSet.Config as cms

# AlCaReco for muon alignment cosmic ray tracks taken during collisions
OutALCARECOMuAlGlobalCosmicsInCollisions_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlGlobalCosmicsInCollisions')
    ),
    outputCommands = cms.untracked.vstring(
	'keep *_ALCARECOMuAlGlobalCosmicsInCollisions_*_*',
	'keep *_ALCARECOMuAlGlobalCosmicsInCollisionsGeneralTracks_*_*', # selected general tracks
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
	'keep DcsStatuss_scalersRawToDigi_*_*',
	'keep Si*Cluster*_si*Clusters_*_*', # for cosmics keep original clusters
	'keep siStripDigis_DetIdCollection_*_*',
	'keep recoMuons_muons1Leg_*_*', # save muons as timing info is needed for BP corrections in deconvolution
    )
)

import copy
OutALCARECOMuAlGlobalCosmicsInCollisions = copy.deepcopy(OutALCARECOMuAlGlobalCosmicsInCollisions_noDrop)
OutALCARECOMuAlGlobalCosmicsInCollisions.outputCommands.insert(0, "drop *")
