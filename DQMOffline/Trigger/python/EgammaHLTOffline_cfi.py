import FWCore.ParameterSet.Config as cms

egammaHLTDQM = cms.EDFilter("EgammaHLTOffline",
                            filters = cms.VPSet(),
                            triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","FU"),
                            EndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
                            BarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
                            DQMDirName=cms.string("HLTOffline/EgammaHLT")
                       
)


