# The following comments couldn't be translated into the new config version:

import FWCore.ParameterSet.Config as cms

# AlCaReco for muon alignment using straight (zero-field) cosmic ray tracks
OutALCARECOMuAlStandAloneCosmics = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlStandAloneCosmics')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
	'keep *_ALCARECOMuAlStandAloneCosmics_*_*',
	'keep *_cosmicMuons_*_*',
	'keep *_cosmictrackfinderP5_*_*',
        'keep Si*Cluster*_*_*_*', # for cosmics keep also clusters
        'keep *_muonCSCDigis_*_*',
	'keep *_muonDTDigis_*_*',
	'keep *_muonRPCDigis_*_*',
	'keep *_dt1DRecHits_*_*',
	'keep *_dt2DSegments_*_*',
	'keep *_dt4DSegments_*_*',
	'keep *_csc2DRecHits_*_*',
	'keep *_cscSegments_*_*',
	'keep *_rpcRecHits_*_*')
)

