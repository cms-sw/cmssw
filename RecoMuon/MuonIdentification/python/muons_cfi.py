# This file name is temporary and ment for development only.
# The content of this file will be moved to muons_cfi as soon as the complete work flow is in place.

import FWCore.ParameterSet.Config as cms

muons = cms.EDProducer("MuonProducer",
                       ActivateDebug = cms.untracked.bool(True),
                       InputMuons = cms.InputTag("muons1stStep"),
                       PFCandidates = cms.InputTag("particleFlowTmp"),
                       
                       EcalIsoDeposits  = cms.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
                       HcalIsoDeposits  = cms.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
                       HoIsoDeposits    = cms.InputTag("muIsoDepositCalByAssociatorTowers","ho"),
                       TrackIsoDeposits = cms.InputTag("muIsoDepositTk"),
                       JetIsoDeposits   = cms.InputTag("muIsoDepositJets"),

                       PFIsolation = cms.PSet(isolationR03 = cms.PSet(chargedParticle = cms.InputTag("muPFIsoValueChargedAll03"),
                                                                      chargedHadron = cms.InputTag("muPFIsoValueCharged03"),
                                                                      neutralHadron = cms.InputTag("muPFIsoValueNeutral03"),
                                                                      photon = cms.InputTag("muPFIsoValueGamma03"),
                                                                      pu = cms.InputTag("muPFIsoValuePU03")),
                                              isolationR04 = cms.PSet(chargedParticle = cms.InputTag("muPFIsoValueChargedAll04"),
                                                                      chargedHadron = cms.InputTag("muPFIsoValueCharged04"),
                                                                      neutralHadron = cms.InputTag("muPFIsoValueNeutral04"),
                                                                      photon = cms.InputTag("muPFIsoValueGamma04"),
                                                                      pu = cms.InputTag("muPFIsoValuePU04"))),

                       FillSelectorMaps = cms.bool(True),
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

                       FillCosmicsIdMap = cms.bool(True),
                       CosmicIdMap = cms.InputTag("cosmicsVeto")
                       )
