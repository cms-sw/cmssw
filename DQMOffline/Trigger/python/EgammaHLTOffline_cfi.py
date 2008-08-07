import FWCore.ParameterSet.Config as cms

egammaHLTDQM = cms.EDFilter("EgammaHLTOffline",
                            filters = cms.VPSet(),
                            triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
                            EndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
                            BarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
                            DQMDirName=cms.string("HLT/EgammaHLTOffline_egammaHLTDQM")
                       
)


