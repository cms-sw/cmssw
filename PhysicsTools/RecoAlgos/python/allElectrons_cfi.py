import FWCore.ParameterSet.Config as cms

allElectrons = cms.EDProducer("GsfElectronCloneProducer",
    src = cms.InputTag("pixelMatchGsfElectrons")
)


# foo bar baz
# DQmRM4urilHUP
# TG7wSjBzy05sU
