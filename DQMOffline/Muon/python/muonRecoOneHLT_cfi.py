import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *

muonRecoOneHLT = cms.EDAnalyzer("MuonRecoOneHLT",
                                MuonServiceProxy,
                                MuonCollection      = cms.InputTag("muons"),
                                VertexLabel         = cms.InputTag("offlinePrimaryVertices"),
                                BeamSpotLabel       = cms.InputTag("offlineBeamSpot"),
                                TriggerResultsLabel = cms.InputTag("TriggerResults::HLT"),
                                
                                ptBin = cms.int32(50),
                                ptMin = cms.double(0.0),
                                ptMax = cms.double(100.0),
                                
                                etaBin = cms.int32(50),
                                etaMin = cms.double(-3.0),
                                etaMax = cms.double(3.0),
                                
                                phiBin = cms.int32(50),
                                phiMin = cms.double(-3.2),
                                phiMax = cms.double(3.2),
                                
                                chi2Bin = cms.int32(50),
                                chi2Min = cms.double(0.),
                                chi2Max = cms.double(20),
                                
                                SingleMuonTrigger = cms.PSet(
                                                             andOr         = cms.bool( True ),
                                                             dbLabel       = cms.string( "MuonDQMTrigger"),
                                                             hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                             hltDBKey      = cms.string('SingleMu'),
                                                             hltPaths      = cms.vstring( "HLT_IsoMu24_eta2p1_v*" ),
                                                             andOrHlt      = cms.bool( True ),
                                                             errorReplyHlt = cms.bool( False ),
                                                             ),
                                DoubleMuonTrigger = cms.PSet(
                                                             andOr         = cms.bool( True ),
                                                             dbLabel       = cms.string( "MuonDQMTrigger"),
                                                             hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                             hltDBKey      = cms.string('DoubleMu'),
                                                             hltPaths      = cms.vstring( "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v*" ),
                                                             andOrHlt      = cms.bool( True ),
                                                             errorReplyHlt = cms.bool( False ),
                                                             )
                                )

