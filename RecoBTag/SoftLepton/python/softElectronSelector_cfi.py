import FWCore.ParameterSet.Config as cms

# select a subset of the GsfElectron collection based on the quality stored in a ValueMap
softElectronSelector = cms.EDProducer("BtagGsfElectronSelector",
    input     = cms.InputTag('gsfElectrons'),
    selection = cms.InputTag('eidLoose'),
    cut       = cms.double(0.5)
)
