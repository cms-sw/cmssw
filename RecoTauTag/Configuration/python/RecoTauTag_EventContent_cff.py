import FWCore.ParameterSet.Config as cms

#Full Event content 
RecoTauTagFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_pfRecoTauTagInfoProducer_*_*', 
        'keep *_fixedConePFTauProducer*_*_*', 
        'keep *_fixedConePFTauDiscrimination*_*_*', 
        'keep *_fixedConeHighEffPFTauProducer*_*_*', 
        'keep *_fixedConeHighEffPFTauDiscrimination*_*_*', 
        'keep *_shrinkingConePFTauProducer*_*_*', 
        'keep *_shrinkingConePFTauDecayModeProducer*_*_*', 
        'keep *_shrinkingConePFTauDecayModeIndexProducer*_*_*', 
        'keep *_shrinkingConePFTauDiscrimination*_*_*', 
        'keep *_caloRecoTauTagInfoProducer_*_*', 
        'keep recoCaloTaus_caloRecoTauProducer*_*_*', 
        'keep *_caloRecoTauDiscrimination*_*_*')
)
#RECO content
RecoTauTagRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_pfRecoTauTagInfoProducer_*_*', 
        'keep *_fixedConePFTauProducer*_*_*', 
        'keep *_fixedConePFTauDiscrimination*_*_*', 
        'keep *_fixedConeHighEffPFTauProducer*_*_*', 
        'keep *_fixedConeHighEffPFTauDiscrimination*_*_*', 
        'keep *_shrinkingConePFTauProducer*_*_*', 
        'keep *_shrinkingConePFTauDecayModeIndexProducer*_*_*', 
        'keep *_shrinkingConePFTauDiscrimination*_*_*', 
        'keep *_caloRecoTauTagInfoProducer_*_*', 
        'keep recoCaloTaus_caloRecoTauProducer*_*_*', 
        'keep *_caloRecoTauDiscrimination*_*_*')
)
#AOD content
RecoTauTagAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_pfRecoTauTagInfoProducer_*_*', 
        'keep *_fixedConePFTauProducer*_*_*', 
        'keep *_fixedConePFTauDiscrimination*_*_*', 
        'keep *_fixedConeHighEffPFTauProducer*_*_*', 
        'keep *_fixedConeHighEffPFTauDiscrimination*_*_*', 
        'keep *_shrinkingConePFTauProducer*_*_*', 
        'keep *_shrinkingConePFTauDecayModeIndexProducer*_*_*', 
        'keep *_shrinkingConePFTauDiscrimination*_*_*', 
        'keep *_caloRecoTauTagInfoProducer_*_*', 
        'keep recoCaloTaus_caloRecoTauProducer*_*_*', 
        'keep *_caloRecoTauDiscrimination*_*_*')
)

