# This file name is temporary and ment for development only.
# The content of this file will be moved to muons_cfi as soon as the complete work flow is in place.

import FWCore.ParameterSet.Config as cms

muons = cms.EDProducer("MuonProducer",
                       ActivateDebug = cms.untracked.bool(False),
                       FastLabelling = cms.untracked.bool(False),
                          
                       InputMuons = cms.InputTag("muons"),

                       FillPFMomentumAndAssociation = cms.bool(True),
                       PFCandidates = cms.InputTag("particleFlow"),
                       
                       FillTimingInfo = cms.bool(True),
                       
                       FillDetectorBasedIsolation = cms.bool(True),
                       EcalIsoDeposits  = cms.InputTag("muons","ecal"),
                       HcalIsoDeposits  = cms.InputTag("muons","hcal"),
                       HoIsoDeposits    = cms.InputTag("muons","ho"),
                       TrackIsoDeposits = cms.InputTag("muons","muIsoDepositTk"),
                       JetIsoDeposits   = cms.InputTag("muons","muIsoDepositJets"),

                       FillPFIsolation = cms.bool(True),                     
                       PFIsolation = cms.PSet(isolationR03 = cms.PSet(chargedParticle = cms.InputTag("muons","muPFIsoValueChargedAll03"),
                                                                      chargedHadron = cms.InputTag("muons","muPFIsoValueCharged03"),
                                                                      neutralHadron = cms.InputTag("muons","muPFIsoValueNeutral03"),
                                                                      photon = cms.InputTag("muons","muPFIsoValueGamma03"),
                                                                      neutralHadronHighThreshold = cms.InputTag("muons","muPFIsoValueNeutralHighThreshold03"),
                                                                      photonHighThreshold = cms.InputTag("muons","muPFIsoValueGammaHighThreshold03"),
                                                                      pu = cms.InputTag("muons","muPFIsoValuePU03")),
                                              isolationR04 = cms.PSet(chargedParticle = cms.InputTag("muons","muPFIsoValueChargedAll04"),
                                                                      chargedHadron = cms.InputTag("muons","muPFIsoValueCharged04"),
                                                                      neutralHadron = cms.InputTag("muons","muPFIsoValueNeutral04"),
                                                                      photon = cms.InputTag("muons","muPFIsoValueGamma04"),
                                                                      neutralHadronHighThreshold = cms.InputTag("muons","muPFIsoValueNeutralHighThreshold04"),
                                                                      photonHighThreshold = cms.InputTag("muons","muPFIsoValueGammaHighThreshold04"),
                                                                      pu = cms.InputTag("muons","muPFIsoValuePU04"))),

                       FillSelectorMaps = cms.bool(True),
                       SelectorMaps = cms.VInputTag(cms.InputTag("muons","muidTMLastStationOptimizedLowPtLoose"),
                                                    cms.InputTag("muons","muidTMLastStationOptimizedLowPtTight"),
                                                    cms.InputTag("muons","muidTM2DCompatibilityLoose"),
                                                    cms.InputTag("muons","muidTM2DCompatibilityTight"),
                                                    cms.InputTag("muons","muidTrackerMuonArbitrated"),
                                                    cms.InputTag("muons","muidTMLastStationAngLoose"),
                                                    cms.InputTag("muons","muidGlobalMuonPromptTight"),
                                                    cms.InputTag("muons","muidGMStaChiCompatibility"),
                                                    cms.InputTag("muons","muidTMLastStationAngTight"),
                                                    cms.InputTag("muons","muidGMTkChiCompatibility"),
                                                    cms.InputTag("muons","muidTMOneStationAngTight"),
                                                    cms.InputTag("muons","muidTMOneStationAngLoose"),
                                                    cms.InputTag("muons","muidTMLastStationLoose"),
                                                    cms.InputTag("muons","muidTMLastStationTight"),
                                                    cms.InputTag("muons","muidTMOneStationTight"),
                                                    cms.InputTag("muons","muidTMOneStationLoose"),
                                                    cms.InputTag("muons","muidAllArbitrated"),
                                                    cms.InputTag("muons","muidGMTkKinkTight"),
                                                    cms.InputTag("muons","muidRPCMuLoose")
                                                    ),

                       FillShoweringInfo = cms.bool(True),
                       ShowerInfoMap = cms.InputTag("muons","muonShowerInformation"),

                       FillCosmicsIdMap = cms.bool(True),
                       CosmicIdMap = cms.InputTag("muons","cosmicsVeto")
                       )
