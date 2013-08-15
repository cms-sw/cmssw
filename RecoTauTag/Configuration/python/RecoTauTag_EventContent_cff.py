import FWCore.ParameterSet.Config as cms

#Full Event content
RecoTauTagFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_ak4PFJetsRecoTauPiZeros_*_*',
        'keep *_hpsPFTauProducer_*_*',
        'keep *_hpsPFTauDiscrimination*_*_*',
      )
)
#RECO content
RecoTauTagRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_ak4PFJetsRecoTauPiZeros_*_*',
        'keep *_hpsPFTauProducer_*_*',
        'keep *_hpsPFTauDiscrimination*_*_*'
        )
)
#AOD content
RecoTauTagAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_ak4PFJetsRecoTauPiZeros_*_*',
        'keep *_hpsPFTauProducer_*_*',
        'keep *_hpsPFTauDiscrimination*_*_*',
        'keep *_hpsTancTaus_*_*',
        'keep *_hpsTancTausDiscrimination*_*_*'
        )
)

