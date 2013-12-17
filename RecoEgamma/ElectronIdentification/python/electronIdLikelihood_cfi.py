import FWCore.ParameterSet.Config as cms

from RecoEgamma.ElectronIdentification.likelihoodPdfsDB_cfi import *
from RecoEgamma.ElectronIdentification.likelihoodESetup_cfi import *
eidLikelihood = cms.EDFilter("EleIdLikelihoodRef",
    filter = cms.bool(False),
    threshold = cms.double(0.5),
    src = cms.InputTag("gedGsfElectrons"),
    reducedBarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
    doLikelihood = cms.bool(True)
)


