import FWCore.ParameterSet.Config as cms

from RecoEgamma.ElectronIdentification.likelihoodPdfsDB_cfi import *
from RecoEgamma.ElectronIdentification.likelihoodESetup_cfi import *
eidLikelihoodExt = cms.EDProducer("EleIdLikelihoodExtProducer",
    src = cms.InputTag("gedGsfElectrons"),
    reducedBarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
    doLikelihood = cms.bool(True)
)


