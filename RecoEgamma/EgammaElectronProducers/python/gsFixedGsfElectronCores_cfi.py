import FWCore.ParameterSet.Config as cms

gsFixedGsfElectronCores = cms.EDProducer("GsfElectronCoreGSCrysFixer",
                                         orgCores=cms.InputTag("gedGsfElectronCores"),
                                         ebRecHits=cms.InputTag("reducedEcalRecHitsEB"), ##ffs, weights dont have the gains switches set properly for some reason, doesnt matter, all we need this for is the gain so we can use orginal multifit
                                         #                             ebRecHits=cms.InputTag("ecalWeightsRecHits","EcalRecHitsEB"), ##ffs, weights dont have the gains switches set properly for some reason
                                         oldRefinedSCToNewMap=cms.InputTag("gsBrokenToGSFixedSuperClustersMap","refinedSCs"),
                                         oldSCToNewMap=cms.InputTag("gsBrokenToGSFixedSuperClustersMap","parentSCs"),
)
