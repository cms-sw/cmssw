import FWCore.ParameterSet.Config as cms

def dropPatTrigger(outputCommands):
    print 'dropping patTrigger'
    outputCommands.append("drop *_*patTrigger*_*_*")
    outputCommands.append("drop *_*PatTrigger*_*_*")

def dropSimDigis(outputCommands):
    outputCommands.append("drop *_sim*Digis*_*_*")
    outputCommands.append("drop *_gmtDigis*_*_*")
 
extraPremixContent = ['keep *_mix_generalTracks_*']

RecoLocalTracker = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_siTrackerGaussianSmearingRecHits_*_*'
        ))

SimRAW = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep edmHepMCProduct_source_*_*',
        'keep *_fastSimProducer_*_*',
        'keep *_MuonSimHits_*_*',
        ))

SimRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep edmHepMCProduct_source_*_*',
        'keep SimTracks_fastSimProducer_*_*',
        'keep SimVertexs_fastSimProducer_*_*',
        ))

FASTPUEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'drop *', 
        'keep *_fastSimProducer_*_*',
        'keep *_MuonSimHits_*_*',
        'drop *_fastSimProducer_VertexTypes_*',    
        'keep *_generalTracksBeforeMixing_*_*',
        'drop *_generalTracksBeforeMixing_MVAValues_*',
        'drop *_generalTracksBeforeMixing_QualityMasks_*',
        'keep edmHepMCProduct_generatorSmeared_*_*'
        ))
