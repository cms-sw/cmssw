import FWCore.ParameterSet.Config as cms

electronFilter = cms.EDFilter("EtaPtMinPixelMatchGsfElectronFullCloneSelector",
    filter = cms.bool(True),
    src = cms.InputTag("gsfElectrons"),
    etaMin = cms.double(-2.5),
    etaMax = cms.double(2.5),
    ptMin = cms.double(15.0)
)


