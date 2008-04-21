import FWCore.ParameterSet.Config as cms

#Full Event content 
RecoTauTagFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_coneIsolationTauJetTags_*_*', 
        'keep *_iterativeCone5PFJets_*_*', 
        'keep *_ic5PFJetTracksAssociatorAtVertex_*_*', 
        'keep *_pfRecoTauTagInfoProducer_*_*', 
        'keep *_pfRecoTauProducer_*_*', 
        'keep *_pfRecoTauDiscriminationByIsolation_*_*', 
        'keep *_pfRecoTauProducerHighEfficiency_*_*', 
        'keep *_pfRecoTauDiscriminationHighEfficiency_*_*', 
        'keep *_caloRecoTauTagInfoProducer_*_*', 
        'keep *_caloRecoTauProducer_*_*', 
        'keep *_caloRecoTauDiscriminationByIsolation_*_*')
)
#RECO content
RecoTauTagRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_coneIsolationTauJetTags_*_*', 
        'keep *_iterativeCone5PFJets_*_*', 
        'keep *_ic5PFJetTracksAssociatorAtVertex_*_*', 
        'keep *_pfRecoTauTagInfoProducer_*_*', 
        'keep *_pfRecoTauProducer_*_*', 
        'keep *_pfRecoTauDiscriminationByIsolation_*_*', 
        'keep *_pfRecoTauProducerHighEfficiency_*_*', 
        'keep *_pfRecoTauDiscriminationHighEfficiency_*_*', 
        'keep *_caloRecoTauTagInfoProducer_*_*', 
        'keep *_caloRecoTauProducer_*_*', 
        'keep *_caloRecoTauDiscriminationByIsolation_*_*')
)
#AOD content
RecoTauTagAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_coneIsolationTauJetTags_*_*', 
        'keep *_iterativeCone5PFJets_*_*', 
        'keep *_ic5PFJetTracksAssociatorAtVertex_*_*', 
        'keep *_pfRecoTauTagInfoProducer_*_*', 
        'keep *_pfRecoTauProducer_*_*', 
        'keep *_pfRecoTauDiscriminationByIsolation_*_*', 
        'keep *_pfRecoTauProducerHighEfficiency_*_*', 
        'keep *_pfRecoTauDiscriminationHighEfficiency_*_*', 
        'keep *_caloRecoTauTagInfoProducer_*_*', 
        'keep *_caloRecoTauProducer_*_*', 
        'keep *_caloRecoTauDiscriminationByIsolation_*_*')
)

