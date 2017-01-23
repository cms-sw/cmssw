import FWCore.ParameterSet.Config as cms

gsFixedGedPhotonCores = cms.EDProducer("GEDPhotonCoreGSCrysFixer",
                                       orgCores=cms.InputTag("gedPhotonCore"),
                                       ebRecHits=cms.InputTag("reducedEcalRecHitsEB"), ##ffs, weights dont have the gains switches set properly for some reason, doesnt matter, all we need this for is the gain so we can use orginal multifit
                                       #                             ebRecHits=cms.InputTag("ecalWeightsRecHits","EcalRecHitsEB"), ##ffs, weights dont have the gains switches set properly for some reason
                                       oldRefinedSCToNewMap=cms.InputTag("gsBrokenToGSFixedSuperClustersMap","refinedSCs"),
                                       oldSCToNewMap=cms.InputTag("gsBrokenToGSFixedSuperClustersMap","parentSCs"),
                                       )
