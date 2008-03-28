import FWCore.ParameterSet.Config as cms

electronFilter = cms.EDFilter("EtaPtMinPixelMatchGsfElectronFullCloneSelector",
    filter = cms.bool(True),
    src = cms.InputTag("pixelMatchGsfElectrons"),
    etaMin = cms.double(-2.7),
    etaMax = cms.double(2.7),
    ptMin = cms.double(5.0)
)


