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
                                          metCollectionTag = cms.untracked.InputTag("hltMet"),
                                          metCleanCollectionTag = cms.untracked.InputTag("hltMetClean"),
                                          numwidejets = cms.uint32(2),
                                          etawidejets = cms.double(2.5),
                                          ptwidejets = cms.double(30),
                                          detawidejets = cms.double(1.3),
                                          dphiwidejets = cms.double(1.0471),# pi/3
                                          maxEMfraction = cms.double(0.95),
                                          maxHADfraction = cms.double(0.95),
                                          HLTpathMain = cms.string("DST_HT250_v*"),
                                          HLTpathMonitor = cms.string("DST_L1HTT_Or_L1MultiJet_v*"),
                                          triggerConfiguration = cms.PSet(
                                              hltResults = cms.InputTag('TriggerResults','','HLT'),
                                              l1tResults = cms.InputTag(''),
                                              daqPartitions = cms.uint32(1),
                                              l1tIgnoreMask = cms.bool( False ),
                                              l1techIgnorePrescales = cms.bool( False ),
                                              throw  = cms.bool( True )
                                          )
                                          )

#this file contains the sequence for data scouting using the DiJet analysis
scoutingDiJetDQMSequence = cms.Sequence(selectedCaloJets*
                                        scoutingDiJetVariables*
                                        scoutingDiJetVarAnalyzer
                                        )
