import FWCore.ParameterSet.Config as cms

# this method replaces outputcommands that apply to normal digis to simdigis
# whenever using this, make sure that this is what you actually want to accomplish
def replaceDigisWithSimDigis(outputCommands):
    for e in range(0,len(outputCommands)):
        pieces = outputCommands[e].split("_")
        if len(pieces) != 4:
            continue
        label = pieces[1]
        pos = label.rfind("Digis")
        if  pos <= 0 or pos != (len(label) - 5):
            continue
        if label.find("sim") == 0:
            continue
        label = "sim"+label[0].upper()+label[1:]
        pieces[1] = label
        outputCommands[e] = "_".join(pieces)
    
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


