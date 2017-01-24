import FWCore.ParameterSet.Config as cms
from RecoEgamma.EgammaTools.regressionModifier_cfi import regressionModifier

gsFixedGsfElectrons = cms.EDProducer("GsfElectronGSCrysFixer",
                                     newCores=cms.InputTag("gsFixedGsfElectronCores"),
                                     oldEles=cms.InputTag("gedGsfElectrons",processName=cms.InputTag.skipCurrentProcess()),
                                     ebRecHits=cms.InputTag("ecalMultiAndGSWeightRecHitEB"),
                                     newCoresToOldCoresMap=cms.InputTag("gsFixedGsfElectronCores","parentCores"),
                                     regressionConfig = regressionModifier.clone(rhoCollection=cms.InputTag("fixedGridRhoFastjetAllTmp")),
                                     
                                     
)
