import FWCore.ParameterSet.Config as cms

gsFixedGedPhotonCores = cms.EDProducer("GEDPhotonCoreGSCrysFixer",
                                       orgCores=cms.InputTag("gedPhotonCore",processName=cms.InputTag.skipCurrentProcess()),
                                       ebRecHits=cms.InputTag("reducedEcalRecHitsEB",processName=cms.InputTag.skipCurrentProcess()), ##ffs, weights dont have the gains switches set properly for some reason, doesnt matter, all we need this for is the gain so we can use orginal multifit
                                       oldRefinedSCToNewMap=cms.InputTag("gsBrokenToGSFixedSuperClustersMap","refinedSCs"),
                                       oldSCToNewMap=cms.InputTag("gsBrokenToGSFixedSuperClustersMap","parentSCs"),
                                       )
