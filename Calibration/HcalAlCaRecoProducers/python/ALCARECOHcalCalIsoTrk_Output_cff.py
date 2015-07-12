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
	'keep *_offlineBeamSpot_*_*',
	'keep edmTriggerResults_*_*_*',
	'keep triggerTriggerEvent_*_*_*',
        'keep recoTracks_generalTracks_*_*',
        'keep recoTrackExtras_generalTracks_*_*',
        )
)


import copy
OutALCARECOHcalCalIsoTrk=copy.deepcopy(OutALCARECOHcalCalIsoTrk_noDrop)
OutALCARECOHcalCalIsoTrk.outputCommands.insert(0, "drop *")
