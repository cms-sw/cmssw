# The following comments couldn't be translated into the new config version:

import FWCore.ParameterSet.Config as cms

# AlCaReco for muon alignment using straight (zero-field) cosmic ray tracks
OutALCARECOMuAlStandAloneCosmics_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlStandAloneCosmics')
    ),
    outputCommands = cms.untracked.vstring(
	'keep *_ALCARECOMuAlStandAloneCosmics_*_*',
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
OutALCARECOMuAlStandAloneCosmics = copy.deepcopy(OutALCARECOMuAlStandAloneCosmics_noDrop)
OutALCARECOMuAlStandAloneCosmics.outputCommands.insert(0, "drop *")
