import FWCore.ParameterSet.Config as cms

#Full Event content ---- temporary
RecoHiJetsRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_*CaloJets_*_*',
                                           'keep *_*PFJets_*_*',
                                           'keep *_*HiGenJets_*_*',
                                           'keep *_*voronoiBackground*_*_*',
                                           'keep *_*PFTowers_*_*'
                                           )
    )

RecoHiJetsFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_*CaloJets_*_*',
                                           'keep *_*PFJets_*_*',
                                           'keep *_*HiGenJets_*_*',
                                           'keep *_*voronoiBackground*_*_*',
                                           'keep *_*PFTowers_*_*'
                                           )
    )

RecoHiJetsAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_*CaloJets_*_*',
                                           'keep *_*PFJets_*_*',
                                           'keep *_*HiGenJets_*_*',
                                           'keep *_*voronoiBackground*_*_*',
                                           'keep *_*PFTowers_*_*',
                                           'keep CaloTowersSorted_towerMaker_*_*',
                                           'drop recoCandidatesOwned_caloTowers_*_*',
                                           'keep recoPFCandidates_particleFlowTmp_*_*'
                                           )
    )




