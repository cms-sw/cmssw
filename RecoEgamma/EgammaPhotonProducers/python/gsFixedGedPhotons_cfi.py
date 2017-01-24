import FWCore.ParameterSet.Config as cms
from RecoEgamma.EgammaTools.regressionModifier_cfi import regressionModifier

gsFixedGedPhotons = cms.EDProducer("GEDPhotonGSCrysFixer",
                                   newCores=cms.InputTag("gsFixedGedPhotonCores"),
                                   oldPhos=cms.InputTag("gedPhotons",processName=cms.InputTag.skipCurrentProcess()),
                                   ebRecHits=cms.InputTag("ecalMultiAndGSWeightRecHitEB"),
                                   newCoresToOldCoresMap=cms.InputTag("gsFixedGedPhotonCores","parentCores"),
                                   regressionConfig = regressionModifier.clone(rhoCollection=cms.InputTag("fixedGridRhoFastjetAllTmp")),
                                   
                                   
                                   )
