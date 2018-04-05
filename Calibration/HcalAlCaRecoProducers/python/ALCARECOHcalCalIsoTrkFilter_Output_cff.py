import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL Isotrk
# output module 
#  module alcastreamHcalIsotrkOutput = PoolOutputModule
OutALCARECOHcalCalIsoTrkFilter_noDrop = cms.PSet(
    # use this in case of filter available
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalIsoTrkFilter')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep *_gtStage2Digis_*_*',
        'keep *_hbheprereco_*_*',
        'keep *_hbhereco_*_*',
        'keep *_ecalRecHit_*_*',
        'keep *_towerMaker_*_*',
        'keep *_offlineBeamSpot_*_*',
        'keep *_hltTriggerSummaryAOD_*_*',
        'keep *_TriggerResults_*_*',
        'keep *_generalTracks_*_*',
        'keep *_generalTracksExtra_*_*',
        'keep *_offlinePrimaryVertices_*_*',
        'keep *_TkAlIsoProdFilter_*_*',
        )
)


import copy
OutALCARECOHcalCalIsoTrkFilter=copy.deepcopy(OutALCARECOHcalCalIsoTrkFilter_noDrop)
OutALCARECOHcalCalIsoTrkFilter.outputCommands.insert(0, "drop *")
