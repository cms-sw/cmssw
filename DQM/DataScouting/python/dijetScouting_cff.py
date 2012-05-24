import FWCore.ParameterSet.Config as cms

selectedCaloJets = cms.EDFilter( "CaloJetSelector",
                                 filter = cms.bool( False ),
                                 src = cms.InputTag( "hltCaloJetIDPassed" ),
                                 cut = cms.string( "abs(eta)<3 && pt>30" ),
                                 #cut = cms.string( "abs(eta)<2.4 && pt>30 && n90 >= 3 && emEnergyFraction > 0.01 && emEnergyFraction < 0.99" )
                                 )

scoutingDiJetVariables = cms.EDProducer("DiJetVarProducer",
                                        inputJetTag = cms.InputTag("selectedCaloJets"),
                                        wideJetDeltaR = cms.double(1.1),
                                        )

scoutingDiJetVarAnalyzer = cms.EDAnalyzer("DiJetVarAnalyzer",
                                          modulePath = cms.untracked.string("DiJet"),
                                          jetCollectionTag = cms.untracked.InputTag("selectedCaloJets"),
                                          #dijetVarCollectionTag = cms.untracked.InputTag("scoutingDiJetVariables","dijetvariables"),
                                          widejetsCollectionTag = cms.untracked.InputTag("scoutingDiJetVariables","widejets"),
                                          numwidejets = cms.uint32(2),
                                          etawidejets = cms.double(2.4),
                                          ptwidejets = cms.double(30),
                                          detawidejets = cms.double(1.3),
                                          dphiwidejets = cms.double(1.0471),# pi/3                                          
                                          )

#this file contains the sequence for data scouting using the DiJet analysis
scoutingDiJetDQMSequence = cms.Sequence(selectedCaloJets*
                                        scoutingDiJetVariables*
                                        scoutingDiJetVarAnalyzer
                                        )
