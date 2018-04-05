import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL Isotrk
# output module 
#  module alcastreamHcalIsotrkOutput = PoolOutputModule
OutALCARECOHcalCalIsoTrk_noDrop = cms.PSet(
    # use this in case of filter available
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalIsoTrk')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep *_IsoProd_*_*',
        'keep *_TkAlIsoProd_*_*',
        'keep *_hbhereco_*_*',
        'keep *_ecalRecHit_*_*',
        'keep *_towerMaker_*_*',
	'keep *_offlineBeamSpot_*_*',
        'keep *_hltTriggerSummaryAOD_*_*',
        'keep *_gtStage2Digis_*_*',
        'keep *_hbheprereco_*_*',
        'keep *_TriggerResults_*_*',
        'keep *_generalTracks_*_*',
        'keep *_generalTracksExtra_*_*',
        'keep *_offlinePrimaryVertices_*_*',
        'keep *_TkAlIsoProdFilter_*_*',
        )
    )

import copy
OutALCARECOHcalCalIsoTrk=copy.deepcopy(OutALCARECOHcalCalIsoTrk_noDrop)
OutALCARECOHcalCalIsoTrk.outputCommands.insert(0, "drop *")
