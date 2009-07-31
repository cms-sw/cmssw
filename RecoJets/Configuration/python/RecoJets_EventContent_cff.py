import FWCore.ParameterSet.Config as cms

#Full Event content 
RecoJetsFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCaloJets_*_*_*', 
        'keep recoPFJets_*_*_*', 
        'keep *_caloTowers_*_*', 
        'keep *_towerMaker_*_*', 
        'keep *_ic5JetTracksAssociatorAtVertex_*_*', 
        'keep *_iterativeCone5JetTracksAssociatorAtVertex_*_*', 
        'keep *_iterativeCone5JetTracksAssociatorAtCaloFace_*_*', 
        'keep *_iterativeCone5JetExtender_*_*', 
        'keep *_sisCone5JetTracksAssociatorAtVertex_*_*', 
        'keep *_sisCone5JetTracksAssociatorAtCaloFace_*_*', 
        'keep *_sisCone5JetExtender_*_*', 
        'keep *_kt4JetTracksAssociatorAtVertex_*_*', 
        'keep *_kt4JetTracksAssociatorAtCaloFace_*_*', 
        'keep *_kt4JetExtender_*_*',
        'keep *_ak5JetTracksAssociatorAtVertex_*_*', 
        'keep *_ak5JetTracksAssociatorAtCaloFace_*_*', 
        'keep *_ak5JetExtender_*_*')
)
RecoGenJetsFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoGenJets_*_*_*', 
        'keep *_genParticle_*_*')
)
#RECO content
RecoJetsRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_kt4CaloJets_*_*', 
        'keep *_kt6CaloJets_*_*',
        'keep *_ak5CaloJets_*_*',
        'keep *_ak7CaloJets_*_*',
        'keep *_iterativeCone5CaloJets_*_*', 
        'keep *_iterativeCone15CaloJets_*_*', 
        'keep *_sisCone5CaloJets_*_*', 
        'keep *_sisCone7CaloJets_*_*', 
        'keep *_kt4PFJets_*_*', 
        'keep *_kt6PFJets_*_*',
        'keep *_ak5PFJets_*_*',
        'keep *_ak7PFJets_*_*',
        'keep *_iterativeCone5PFJets_*_*', 
        'keep *_sisCone5PFJets_*_*', 
        'keep *_sisCone7PFJets_*_*', 
        'keep *_caloTowers_*_*', 
        'keep *_towerMaker_*_*', 
        'keep *_ic5JetTracksAssociatorAtVertex_*_*', 
        'keep *_iterativeCone5JetTracksAssociatorAtVertex_*_*', 
        'keep *_iterativeCone5JetTracksAssociatorAtCaloFace_*_*', 
        'keep *_iterativeCone5JetExtender_*_*', 
        'keep *_sisCone5JetTracksAssociatorAtVertex_*_*', 
        'keep *_sisCone5JetTracksAssociatorAtCaloFace_*_*', 
        'keep *_sisCone5JetExtender_*_*', 
        'keep *_kt4JetTracksAssociatorAtVertex_*_*', 
        'keep *_kt4JetTracksAssociatorAtCaloFace_*_*', 
        'keep *_kt4JetExtender_*_*',
        'keep *_ak5JetTracksAssociatorAtVertex_*_*', 
        'keep *_ak5JetTracksAssociatorAtCaloFace_*_*', 
        'keep *_ak5JetExtender_*_*')
)
RecoGenJetsRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_kt4GenJets_*_*', 
        'keep *_kt6GenJets_*_*',
        'keep *_ak5GenJets_*_*',
        'keep *_ak7GenJets_*_*',
        'keep *_iterativeCone5GenJets_*_*', 
        'keep *_sisCone5GenJets_*_*', 
        'keep *_sisCone7GenJets_*_*', 
        'keep *_genParticle_*_*')
)
#AOD content
RecoJetsAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_kt4CaloJets_*_*', 
        'keep *_kt6CaloJets_*_*',
        'keep *_ak5CaloJets_*_*',
        'keep *_ak7CaloJets_*_*',
        'keep *_iterativeCone5CaloJets_*_*', 
        'keep *_iterativeCone15CaloJets_*_*', 
        'keep *_sisCone5CaloJets_*_*', 
        'keep *_sisCone7CaloJets_*_*',  
        'keep *_kt4PFJets_*_*', 
        'keep *_kt6PFJets_*_*',
        'keep *_ak5PFJets_*_*',
        'keep *_iterativeCone5PFJets_*_*', 
        'keep *_sisCone5PFJets_*_*', 
        'keep *_sisCone7PFJets_*_*', 
        'keep *_caloTowers_*_*', 
        'keep *_towerMaker_*_*', 
        'keep *_ic5JetTracksAssociatorAtVertex_*_*',
        'keep *_sisCone5JetTracksAssociatorAtVertex_*_*', 
        'keep *_ak5JetTracksAssociatorAtVertex_*_*', 
        'keep *_iterativeCone5JetExtender_*_*', 
        'keep *_sisCone5JetExtender_*_*', 
        'keep *_kt4JetExtender_*_*', 
        'keep *_ak5JetExtender_*_*')
)
RecoGenJetsAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_kt4GenJets_*_*', 
        'keep *_kt6GenJets_*_*',
        'keep *_ak5GenJets_*_*',
        'keep *_ak7GenJets_*_*',
        'keep *_iterativeCone5GenJets_*_*', 
        'keep *_sisCone5GenJets_*_*', 
        'keep *_sisCone7GenJets_*_*', 
        'keep *_genParticle_*_*')
)
