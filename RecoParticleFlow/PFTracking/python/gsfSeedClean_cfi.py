import FWCore.ParameterSet.Config as cms

gsfSeedclean = cms.EDProducer("GsfSeedCleaner",
    PreIdSeedLabel = cms.InputTag("elecpreid","SeedsForGsf"),
    TkColList = cms.VInputTag(cms.InputTag("pixelMatchGsfElectrons"))
)


