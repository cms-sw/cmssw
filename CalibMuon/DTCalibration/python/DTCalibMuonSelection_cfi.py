import FWCore.ParameterSet.Config as cms

DTCalibMuonSelection = cms.EDFilter("DTCalibMuonSelection",
    filter = cms.bool(True),
    src = cms.InputTag("muons"),
    etaMin = cms.double(-1.25),
    etaMax = cms.double(1.25),
    ptMin = cms.double(3.)
)
# foo bar baz
# v1e6AgTMx8HZ9
# Bm2DAxUKOxvN7
