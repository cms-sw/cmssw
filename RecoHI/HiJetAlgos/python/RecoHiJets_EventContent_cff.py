import FWCore.ParameterSet.Config as cms

#Full Event content
RecoHiJetsRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCaloJets_iterativeConePu5CaloJets_*_*'
                                           )
    )

RecoHiJetsFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCaloJets_iterativeConePu5CaloJets_*_*',
                                           'keep recoGenJets_iterativeCone5HiGenJets_*_*'
                                           )
    )




