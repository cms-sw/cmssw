import FWCore.ParameterSet.Config as cms
from RecoEgamma.EgammaTools.regressionModifier_cfi import regressionModifier

gedGsfElectronsGSFixed = cms.EDProducer("GsfElectronGSCrysFixer",
    newCores=cms.InputTag("gedGsfElectronCoresGSFixed"),
    oldEles=cms.InputTag("gedGsfElectrons", '', cms.InputTag.skipCurrentProcess()),
    ebRecHits=cms.InputTag("ecalMultiAndGSGlobalRecHitEB"),
    regressionConfig = regressionModifier.clone(rhoCollection=cms.InputTag("fixedGridRhoFastjetAllTmp")),
)
