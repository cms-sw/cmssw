import FWCore.ParameterSet.Config as cms

isolEcalPixelTrackProd = cms.EDProducer("IsolatedEcalPixelTrackCandidateProducer",
                                        filterLabel               = cms.InputTag("hltIsolPixelTrackL2Filter"),
                                        EBRecHitSource            = cms.InputTag("hltEcalRecHit", "EcalRecHitsEB"),
                                        EERecHitSource            = cms.InputTag("hltEcalRecHit", "EcalRecHitsEE"),
                                        ECHitEnergyThreshold      = cms.double(0.05),
                                        ECHitCountEnergyThreshold = cms.double(0.5),
                                        EcalConeSizeEta0          = cms.double(0.09),
                                        EcalConeSizeEta1          = cms.double(0.14)
                                        )


