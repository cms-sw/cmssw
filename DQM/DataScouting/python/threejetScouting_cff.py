import FWCore.ParameterSet.Config as cms

scoutingThreeJetVariables = cms.EDProducer("ThreeJetVarProducer",
                                        inputJetTag = cms.InputTag("hltCaloJetIDPassed"),
                                        )
scoutingThreeJetVarAnalyzer = cms.EDAnalyzer("ThreeJetVarAnalyzer",
                                             modulePath=cms.untracked.string("ThreeJet"),
                                             jetPtCollectionTag    = cms.untracked.InputTag("scoutingThreeJetVariables","jetPt"),
                                             tripPtCollectionTag   = cms.untracked.InputTag("scoutingThreeJetVariables","tripSumPt"),
                                             tripMassCollectionTag = cms.untracked.InputTag("scoutingThreeJetVariables","tripMass"),
                                             jetPtCut      = cms.double(40.0),
                                             htCut         = cms.double(250.0),
                                             delta         = cms.double(110.0)
                                             )

#this file contains the sequence for data scouting using the ThreeJet analysis
scoutingThreeJetDQMSequence = cms.Sequence(scoutingThreeJetVariables*
                                           scoutingThreeJetVarAnalyzer)
