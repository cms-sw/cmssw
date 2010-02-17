import FWCore.ParameterSet.Config as cms

DTCalibMuonSelection = cms.EDFilter("DTCalibMuonSelection",
    filter = cms.bool(True),
    src = cms.InputTag("muons"),
    etaMin = cms.double(-2.5),
    etaMax = cms.double(2.5),
    ptMin = cms.double(0.0)
)
