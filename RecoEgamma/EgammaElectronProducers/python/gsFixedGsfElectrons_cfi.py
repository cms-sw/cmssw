import FWCore.ParameterSet.Config as cms
from RecoEgamma.EgammaTools.regressionModifier_cfi import regressionModifier

gsFixedGsfElectrons = cms.EDProducer("GsfElectronGSCrysFixer",
    newCores=cms.InputTag("gsFixedGsfElectronCores"),
    oldEles=cms.InputTag("gedGsfElectrons", '', cms.InputTag.skipCurrentProcess()),
    ebRecHits=cms.InputTag("ecalMultiAndGSGlobalRecHitEB"),
    regressionConfig = regressionModifier.clone(rhoCollection=cms.InputTag("fixedGridRhoFastjetAllTmp")),
)
