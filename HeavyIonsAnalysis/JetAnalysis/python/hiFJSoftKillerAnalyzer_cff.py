import FWCore.ParameterSet.Config as cms

hiFJSoftKillerAnalyzer = cms.EDAnalyzer(
    'HiFJSoftKillerAnalyzer',
    source = cms.InputTag("packedPFCandidates"),
    etaMap = cms.vdouble(-5., -4., -3, -2.5, -2.0, -0.8, 0.8, 2.0, 2.5, 3., 4., 5., -5., 5.)
)
