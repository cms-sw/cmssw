import FWCore.ParameterSet.Config as cms

hltTauExtTable = cms.EDProducer("HLTTauTableProducer",
                                skipNonExistingSrc = cms.bool(True),
                                taus = cms.InputTag( "hltHpsPFTauProducer" ),
                                deepTauVSe = cms.InputTag("hltHpsPFTauProducer", "VSe"),
                                deepTauVSmu = cms.InputTag("hltHpsPFTauProducer", "VSmu"),
                                deepTauVSjet = cms.InputTag("hltHpsPFTauProducer", "VSjet"),
                                tauTransverseImpactParameters = cms.InputTag( "hltHpsPFTauTransverseImpactParametersForDeepTau" ),
                                precision = cms.int32(7),
                                )
