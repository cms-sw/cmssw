import FWCore.ParameterSet.Config as cms

allElectrons = cms.EDProducer("GsfElectronShallowCloneProducer",
    src = cms.InputTag("pixelMatchGsfElectrons")
)


