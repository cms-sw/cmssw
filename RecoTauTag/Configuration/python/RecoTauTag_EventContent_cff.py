import FWCore.ParameterSet.Config as cms

#Full Event content
RecoTauTagFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_pfRecoTauTagInfoProducer_*_*',
        'keep *_ak5PFJetsRecoTauPiZeros_*_*',
        'keep *_hpsPFTauProducer*_*_*',
        'keep *_hpsPFTauDiscrimination*_*_*',
        'keep *_shrinkingConePFTauProducer*_*_*',
        'keep *_shrinkingConePFTauDiscrimination*_*_*',
        'keep *_hpsTancTaus_*_*',
        'keep *_hpsTancTausDiscrimination*_*_*',
	    'keep *_TCTauJetPlusTrackZSPCorJetAntiKt5_*_*',
        'keep *_caloRecoTauTagInfoProducer_*_*',
        'keep recoCaloTaus_caloRecoTauProducer*_*_*',
        'keep *_caloRecoTauDiscrimination*_*_*')
)
#RECO content
RecoTauTagRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_pfRecoTauTagInfoProducer_*_*',
        'keep *_ak5PFJetsRecoTauPiZeros_*_*',
        'keep *_hpsPFTauProducer*_*_*',
        'keep *_hpsPFTauDiscrimination*_*_*',
        'keep *_shrinkingConePFTauProducer*_*_*',
        'keep *_shrinkingConePFTauDiscrimination*_*_*',
        'keep *_hpsTancTaus_*_*',
        'keep *_hpsTancTausDiscrimination*_*_*',
        'keep *_TCTauJetPlusTrackZSPCorJetAntiKt5_*_*',
        'keep *_caloRecoTauTagInfoProducer_*_*',
        'keep recoCaloTaus_caloRecoTauProducer*_*_*',
        'keep *_caloRecoTauDiscrimination*_*_*')
)
#AOD content
RecoTauTagAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_pfRecoTauTagInfoProducer_*_*',
        'keep *_ak5PFJetsRecoTauPiZeros_*_*',
        'keep *_hpsPFTauProducer*_*_*',
        'keep *_hpsPFTauDiscrimination*_*_*',
        'keep *_shrinkingConePFTauProducer*_*_*',
        'keep *_shrinkingConePFTauDiscrimination*_*_*',
        'keep *_hpsTancTaus_*_*',
        'keep *_hpsTancTausDiscrimination*_*_*',
	    'keep *_TCTauJetPlusTrackZSPCorJetAntiKt5_*_*',
        'keep *_caloRecoTauTagInfoProducer_*_*',
        'keep recoCaloTaus_caloRecoTauProducer*_*_*',
        'keep *_caloRecoTauDiscrimination*_*_*')
)

