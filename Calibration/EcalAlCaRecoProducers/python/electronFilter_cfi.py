import FWCore.ParameterSet.Config as cms

electronFilter = cms.EDFilter("EtaPtMinGsfElectronFullCloneSelector",
    filter = cms.bool(True),
    src = cms.InputTag("gedGsfElectrons"),
    etaMin = cms.double(-2.5),
    etaMax = cms.double(2.5),
    ptMin = cms.double(15.0)
)


