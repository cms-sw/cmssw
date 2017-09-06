import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL Isotrk
# output module 
#  module alcastreamHcalIsotrkOutput = PoolOutputModule
OutALCARECOHcalCalIsoTrkNoHLT_noDrop = cms.PSet(
    # use this in case of filter available
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalIsoTrkNoHLT')
        ),
    outputCommands = cms.untracked.vstring( 
        'keep *_IsoProd_*_*',
        'keep *_TkAlIsoProd_*_*',
	'keep *_offlineBeamSpot_*_*',
        'keep recoTracks_generalTracks_*_*',
        'keep recoTrackExtras_generalTracks_*_*',
        'keep *_gtStage2Digis_*_*',
        'keep *_hbheprereco_*_*',
        'keep edmTriggerResults_*_*_*',
        'keep triggerTriggerEvent_*_*_*',
        )
)


import copy
OutALCARECOHcalCalIsoTrkNoHLT=copy.deepcopy(OutALCARECOHcalCalIsoTrkNoHLT_noDrop)
OutALCARECOHcalCalIsoTrkNoHLT.outputCommands.insert(0, "drop *")
