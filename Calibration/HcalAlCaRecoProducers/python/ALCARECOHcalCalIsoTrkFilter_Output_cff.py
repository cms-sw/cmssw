import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL Isotrk
# output module 
#  module alcastreamHcalIsotrkOutput = PoolOutputModule
OutALCARECOHcalCalIsoTrkFilter_noDrop = cms.PSet(
    # use this in case of filter available
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalIsoTrk')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep *_hbhereco_*_*',
        'keep *_reducedHcalRecHits_*_*',
        'keep *_ecalRecHit_*_*',
	'keep *_offlineBeamSpot_*_*',
        'keep *_hltTriggerSummaryAOD_*_HLT',
        'keep *_TriggerResults_*_*',
	'keep edmTriggerResults_*_*_*',
	'keep triggerTriggerEvent_*_*_*',
        'keep *_generalTracks_*_*',
        'keep *_generalTracks_*_*',
        'keep *_offlinePrimaryVertices_*_*',
        'keep *_TkAlIsoProd_*_*',
        )
)


import copy
OutALCARECOHcalCalIsoTrkFilter=copy.deepcopy(OutALCARECOHcalCalIsoTrkFilter_noDrop)
OutALCARECOHcalCalIsoTrkFilter.outputCommands.insert(0, "drop *")
