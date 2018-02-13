import FWCore.ParameterSet.Config as cms

dropPatTrigger = [
    "drop *_*patTrigger*_*_*",
    "drop *_*PatTrigger*_*_*"
]

dropSimDigis = [
    "drop *_sim*Digis*_*_*",
    "drop *_gmtDigis*_*_*"
]
 
extraPremixContent = ['keep *_mix_generalTracks_*']

RecoLocalTracker = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_siTrackerGaussianSmearingRecHits_*_*'
        ))

SimRAW = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep edmHepMCProduct_source_*_*',
        'keep *_famosSimHits_*_*',
        'keep *_MuonSimHits_*_*',
        ))

SimRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep edmHepMCProduct_source_*_*',
        'keep SimTracks_famosSimHits_*_*',
        'keep SimVertexs_famosSimHits_*_*',
        ))

FASTPUEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'drop *', 
        'keep *_famosSimHits_*_*',
        'keep *_MuonSimHits_*_*',
        'drop *_famosSimHits_VertexTypes_*',    
        'keep *_generalTracksBeforeMixing_*_*',
        'drop *_generalTracksBeforeMixing_MVAValues_*',
        'drop *_generalTracksBeforeMixing_QualityMasks_*',
        'keep edmHepMCProduct_generatorSmeared_*_*'
        ))
