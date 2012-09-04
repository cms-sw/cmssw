import FWCore.ParameterSet.Config as cms

scoutingDiJetPairsVariables = cms.EDProducer("DiJetPairsVarProducer",
                                        inputJetTag = cms.InputTag("hltCaloJetIDPassed"),
                                        )
scoutingDiJetPairsVarAnalyzer = cms.EDAnalyzer("DiJetPairsVarAnalyzer",
                                             modulePath=cms.untracked.string("DiJetPairs"),
                                             jetPtCollectionTag     = cms.untracked.InputTag("scoutingDiJetPairsVariables","jetPt"),
                                             dijetMassCollectionTag = cms.untracked.InputTag("scoutingDiJetPairsVariables","dijetMass"),
                                             dijetPtCollectionTag   = cms.untracked.InputTag("scoutingDiJetPairsVariables","dijetSumPt"),
                                             dijetdRCollectionTag   = cms.untracked.InputTag("scoutingDiJetPairsVariables","dijetdRjj"),
                                             jetPtCut      = cms.double(50.0),
                                             htCut         = cms.double(300.0),
                                             delta         = cms.double(25.0)
                                             )

#this file contains the sequence for data scouting using the DiJetPairs analysis
scoutingDiJetPairsDQMSequence = cms.Sequence(scoutingDiJetPairsVariables*
                                           scoutingDiJetPairsVarAnalyzer)
