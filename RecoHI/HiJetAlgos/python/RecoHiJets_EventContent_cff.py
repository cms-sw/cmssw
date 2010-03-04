import FWCore.ParameterSet.Config as cms

#Full Event content ---- temporary
RecoHiJetsRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCaloJets_*Pu*CaloJets_*_*',
                                           'keep recoGenJets_*HiGenJets_*_*'
                                           )
    )

RecoHiJetsFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCaloJets_*Pu*CaloJets_*_*',
                                           'keep recoGenJets_*HiGenJets_*_*'
                                           )
    )




