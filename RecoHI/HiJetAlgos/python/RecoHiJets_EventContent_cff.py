import FWCore.ParameterSet.Config as cms

#Full Event content ---- temporary
RecoHiJetsRECO = cms.PSet(
    outputCommands = cms.untracked.vstring( 'keep *_akPu3CaloJets_*_*',
                                            'keep *_akPu4CaloJets_*_*', 
                                            'keep *_akPu5CaloJets_*_*', 
                                            'keep *_iterativeConePu5CaloJets_*_*', 
                                            'keep *_akPu3PFJets_*_*',
                                            'keep *_akPu4PFJets_*_*',
                                            'keep *_akPu5PFJets_*_*',
                                            'keep *_akCs3PFJets_*_*',
                                            'keep *_akCs4PFJets_*_*',
                                            'keep *_kt4PFJetsForRho_*_*',
                                            'keep *_*HiGenJets_*_*',
                                            'keep *_PFTowers_*_*',
                                            'keep *_hiFJRhoProducer_*_*'

                                           )
    )

RecoHiJetsFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_*CaloJets_*_*',
                                           'keep *_*PFJets_*_*',
                                           'keep *_*HiGenJets_*_*',
                                           'keep *_*PFTowers_*_*',
                                           'keep *_*hiFJRhoProducer_*_*'
                                           )
    )

RecoHiJetsAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_*CaloJets_*_*',
                                           'keep *_*PFJets_*_*',
                                           'keep *_*HiGenJets_*_*',
                                           'keep *_*PFTowers_*_*',
                                           'keep *_*hiFJRhoProducer_*_*',
                                           'keep CaloTowersSorted_towerMaker_*_*',
                                           'drop recoCandidatesOwned_caloTowers_*_*',
                                           'keep recoPFCandidates_particleFlowTmp_*_*'
                                           )
    )




