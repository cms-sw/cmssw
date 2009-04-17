import FWCore.ParameterSet.Config as cms

egammaHLTDQM = cms.EDFilter("EgammaHLTOffline",
                            filters = cms.VPSet(),
                            triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
                            EndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
                            BarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
                            CaloJetCollection = cms.InputTag("iterativeCone5CaloJets"),
                            DQMDirName=cms.string("HLT/EgammaHLTOffline_egammaHLTDQM"),
                            eleHLTPathNames=cms.vstring("hltL1NonIsoHLTNonIsoSingleElectronEt15",
                                                  "hltL1NonIsoHLTNonIsoSingleElectronLWEt15"),
                            eleHLTFilterNames=cms.vstring("TrackIsolFilter")
                                                         
)


