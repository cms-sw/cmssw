import FWCore.ParameterSet.Config as cms

#Full Event content
RecoHiJetsFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoJets_iterativeConePu5CaloJets_*_*',
                                           'keep recoJets_iterativeCone5HiGenJets_*_*'
                                           )
    )





