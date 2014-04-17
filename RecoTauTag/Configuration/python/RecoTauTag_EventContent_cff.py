import FWCore.ParameterSet.Config as cms

#Full Event content
RecoTauTagFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_ak5PFJetsRecoTauPiZeros_*_*',
        'keep *_hpsPFTauProducer_*_*',
        'keep *_hpsPFTauDiscrimination*_*_*',
        'keep *_hpsPFTau*PtSum_*_*',
        'keep *_hpsPFTauTransverseImpactParameters_*_*'
    )
)
#RECO content
RecoTauTagRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_ak5PFJetsRecoTauPiZeros_*_*',
        'keep *_hpsPFTauProducer_*_*',
        'keep *_hpsPFTauDiscrimination*_*_*',
        'keep *_hpsPFTau*PtSum_*_*',
        'keep *_hpsPFTauTransverseImpactParameters_*_*'
    )
)
#AOD content
RecoTauTagAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_ak5PFJetsRecoTauPiZeros_*_*',
        'keep *_hpsPFTauProducer_*_*',
        'keep *_hpsPFTauDiscrimination*_*_*',
        'keep *_hpsPFTau*PtSum_*_*',
        'keep *_hpsPFTauTransverseImpactParameters_*_*'
    )
)

