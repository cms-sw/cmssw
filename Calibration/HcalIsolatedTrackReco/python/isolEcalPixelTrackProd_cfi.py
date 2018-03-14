import FWCore.ParameterSet.Config as cms

isolEcalPixelTrackProd = cms.EDProducer("IsolatedEcalPixelTrackCandidateProducer",
                                        filterLabel               = cms.InputTag("hltIsolPixelTrackL2Filter"),
                                        EBRecHitSource            = cms.InputTag("hltEcalRecHit", "EcalRecHitsEB"),
                                        EERecHitSource            = cms.InputTag("hltEcalRecHit", "EcalRecHitsEE"),
                                        EBHitEnergyThreshold      = cms.double(0.10),
                                        EBHitCountEnergyThreshold = cms.double(0.5),
                                        EEHitEnergyThreshold0     = cms.double(-20.5332),
                                        EEHitEnergyThreshold1     = cms.double(34.3975),
                                        EEHitEnergyThreshold2     = cms.double(-19.0741),
                                        EEHitEnergyThreshold3     = cms.double(3.52151),
                                        EEFacHitCountEnergyThreshold= cms.double(10.0),
                                        EcalConeSizeEta0          = cms.double(0.09),
                                        EcalConeSizeEta1          = cms.double(0.14)
                                        )


