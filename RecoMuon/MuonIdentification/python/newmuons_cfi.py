import FWCore.ParameterSet.Config as cms

newmuons = cms.EDProducer("MuonProducer",
                          ActivateDebug = cms.untracked.bool(True),
                          InputMuons = cms.InputTag("muons"),
                          PFCandidates = cms.InputTag("particleFlow"),
                          
                          EcalIsoDeposits  = cms.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
                          HcalIsoDeposits  = cms.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
                          HoIsoDeposits    = cms.InputTag("muIsoDepositCalByAssociatorTowers","ho"),
                          TrackIsoDeposits = cms.InputTag("muIsoDepositTk"),
                          JetIsoDeposits   = cms.InputTag("muIsoDepositJets"),
                          
                          SelectorMaps = cms.VInputTag(cms.InputTag("muidTMLastStationOptimizedLowPtLoose"),
                                                       cms.InputTag("muidTMLastStationOptimizedLowPtTight"),
                                                       cms.InputTag("muidTM2DCompatibilityLoose"),
                                                       cms.InputTag("muidTM2DCompatibilityTight"),
                                                       cms.InputTag("muidTrackerMuonArbitrated"),
                                                       cms.InputTag("muidTMLastStationAngLoose"),
                                                       cms.InputTag("muidGlobalMuonPromptTight"),
                                                       cms.InputTag("muidGMStaChiCompatibility"),
                                                       cms.InputTag("muidTMLastStationAngTight"),
                                                       cms.InputTag("muidGMTkChiCompatibility"),
                                                       cms.InputTag("muidTMOneStationAngTight"),
                                                       cms.InputTag("muidTMOneStationAngLoose"),
                                                       cms.InputTag("muidTMLastStationLoose"),
                                                       cms.InputTag("muidTMLastStationTight"),
                                                       cms.InputTag("muidTMOneStationTight"),
                                                       cms.InputTag("muidTMOneStationLoose"),
                                                       cms.InputTag("muidAllArbitrated"),
                                                       cms.InputTag("muidGMTkKinkTight")
                                                       ),
                          
                          ShowerInfoMap = cms.InputTag("muonShowerInformation"),
                          CosmicIdMap = cms.InputTag("cosmicsVeto")
                          
                          )
