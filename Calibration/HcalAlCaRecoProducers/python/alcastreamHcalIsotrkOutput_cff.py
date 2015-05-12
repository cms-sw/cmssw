import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL Isotrk
# output module 
#  module alcastreamHcalIsotrkOutput = PoolOutputModule
alcastreamHcalIsotrkOutput = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
                                           'keep *_offlineBeamSpot_*_*',
                                           'keep edmTriggerResults_*_*_*',
                                           'keep triggerTriggerEvent_*_*_*',
                                           'keep recoTracks_generalTracks_*_*',
                                           'keep recoTrackExtras_generalTracks_*_*',
                                           'keep *_IsoProd_*_*',
                                           )
)

