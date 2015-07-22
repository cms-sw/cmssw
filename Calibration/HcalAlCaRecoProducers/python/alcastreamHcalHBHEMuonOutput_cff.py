import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL HBHEMuon
# output module 
#  module alcastreamHcalHBHEMuonOutput = PoolOutputModule
alcastreamHcalHBHEMuonOutput = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
                                           'keep edmTriggerResults_*_*_*',
                                           'keep recoTracks_globalMuons_*_*',
                                           'keep recoTrackExtras_globalMuons_*_*',
                                           'keep recoTracks_standAloneMuons_*_*',
                                           'keep recoTrackExtras_standAloneMuons_*_*', 
                                           'keep recoTracks_generalTracks_*_*',
                                           'keep recoTrackExtras_generalTracks_*_*',
                                           'keep recoTracks_tevMuons_*_*',
                                           'keep recoTrackExtras_tevMuons_*_*',
                                           'keep *_HBHEMuonProd_*_*')
    )
