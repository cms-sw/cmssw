import FWCore.ParameterSet.Config as cms

allElectrons = cms.EDProducer("GsfElectronCloneProducer",
    src = cms.InputTag("pixelMatchGsfElectrons")
)


