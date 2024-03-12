import FWCore.ParameterSet.Config as cms

allElectrons = cms.EDProducer("GsfElectronShallowCloneProducer",
    src = cms.InputTag("pixelMatchGsfElectrons")
)


# foo bar baz
# saUKOnkTX1fEk
