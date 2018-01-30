import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
AnalyzeSUSYDQM = DQMEDAnalyzer('SUSYDQMAnalyzer',
                                folderName = cms.string("JetMET/SUSYDQM/"),
                                PFMETCollectionLabel   = cms.InputTag("pfMet"),
                                CaloMETCollectionLabel   = cms.InputTag("caloMet"),
				#TCMETCollectionLabel   = cms.InputTag("tcMet"),
				CaloJetCollectionLabel = cms.InputTag("ak4CaloJets"),
				#JPTJetCollectionLabel = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5"),
                                PFJetCollectionLabel = cms.InputTag("ak4PFJets"),
				ptThreshold = cms.double(20.),
                                maxNJets = cms.double(10),
                                maxAbsEta = cms.double(2.4),
                                JetTrigger = cms.PSet(
                                    andOr         = cms.bool( False ),
                                    hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
                                    hltDBKey       = cms.string( 'jetmet_lowptjet' ),
                                    hltPaths       = cms.vstring( 'HLT_L1Jet6U' ), 
                                    andOrHlt       = cms.bool( False ),
                                    errorReplyHlt  = cms.bool( False ),
                                    )                                
                                )
