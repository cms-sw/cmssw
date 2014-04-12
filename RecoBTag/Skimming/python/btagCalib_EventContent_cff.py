import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
# in case we want only the tagInfos and nothing more:
# (this means we need the jets for pt, eta and the JTA for being able
#  to find the jet ref from the tag info)
BTAGCALAbtagCalibEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_mcAlgoJetFlavour_*_*', 
        'keep *_mcPhysJetFlavour_*_*', 
        'keep *_iterativeCone5CaloJets_*_*', 
        'keep *_jetTracksAssociator_*_*', 
        'keep *_impactParameterTagInfos_*_*', 
        'keep *_combinedSVTagInfos_*_*')
)
# in case we want to be able to compute the TagInfos ourselves
# (basically we need tracks and primary vertices for that)
BTAGCALBbtagCalibEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_mcAlgoJetFlavour_*_*', 
        'keep *_mcPhysJetFlavour_*_*', 
        'keep *_iterativeCone5CaloJets_*_*', 
        'keep *_ctfWithMaterialTracks_*_*', 
        'keep *_offlinePrimaryVerticesFromCTFTracks_*_*')
)
btagCalibEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('btagCalibPath')
    )
)

