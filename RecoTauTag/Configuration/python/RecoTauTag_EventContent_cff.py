import FWCore.ParameterSet.Config as cms

#AOD content
RecoTauTagAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep recoRecoTauPiZeros_hpsPFTauProducer_pizeros_*',
        'keep recoPFTaus_hpsPFTauProducer_*_*',
        'keep *_hpsPFTauBasicDiscriminators_*_*',
        'keep *_hpsPFTauBasicDiscriminatorsdR03_*_*',
        'keep *_hpsPFTauDiscriminationByDeadECALElectronRejection_*_*',
        'keep *_hpsPFTauDiscriminationByDecayModeFinding_*_*',
        'keep *_hpsPFTauDiscriminationByDecayModeFindingNewDMs_*_*',
        'keep *_hpsPFTauDiscriminationByDecayModeFindingOldDMs_*_*',
        'keep *_hpsPFTauDiscriminationByMuonRejection3_*_*',
        'keep *_hpsPFTauTransverseImpactParameters_*_*',
        'keep *_hpsPFTauDiscriminationByMVA6ElectronRejection_*_*',
        'keep *_hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLT_*_*',
        'keep *_hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLT_*_*',
        'keep *_hpsPFTauDiscriminationByIsolationMVArun2v1PWoldDMwLT_*_*',
        'keep *_hpsPFTauDiscriminationByIsolationMVArun2v1PWnewDMwLT_*_*',
        'keep *_hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLT_*_*',
        'keep *_hpsPFTauDiscriminationByIsolationMVArun2v1PWdR03oldDMwLT_*_*')
)

#RECO content
RecoTauTagRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_hpsPFTauDiscriminationByLooseElectronRejection_*_*')
)
RecoTauTagRECO.outputCommands.extend(RecoTauTagAOD.outputCommands)

#Full Event content
RecoTauTagFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_ak4PFJetsRecoTauPiZeros_*_*',
        'keep *_hpsPFTauProducer_*_*',
        'keep *_hpsPFTauDiscrimination*_*_*',
        'keep *_hpsPFTau*PtSum_*_*')
)
RecoTauTagFEVT.outputCommands.extend(RecoTauTagRECO.outputCommands)
