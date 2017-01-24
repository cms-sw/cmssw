import FWCore.ParameterSet.Config as cms

gsFixedGsfElectronCores = cms.EDProducer("GsfElectronCoreGSCrysFixer",
                                         orgCores=cms.InputTag("gedGsfElectronCores",processName=cms.InputTag.skipCurrentProcess()),
                                         ebRecHits=cms.InputTag("reducedEcalRecHitsEB",processName=cms.InputTag.skipCurrentProcess()), ##ffs, weights dont have the gains switches set properly for some reason, doesnt matter, all we need this for is the gain so we can use orginal multifit
                                         #                             ebRecHits=cms.InputTag("ecalWeightsRecHits","EcalRecHitsEB"), ##ffs, weights dont have the gains switches set properly for some reason
                                         oldRefinedSCToNewMap=cms.InputTag("gsBrokenToGSFixedSuperClustersMap","refinedSCs"),
                                         oldSCToNewMap=cms.InputTag("gsBrokenToGSFixedSuperClustersMap","parentSCs"),
)
