import FWCore.ParameterSet.Config as cms

tauTo3MuFilter = cms.EDFilter("Tau3MuSkim",
    RecoAnalysisMuonMass = cms.double(0.1057),
    RecoAnalysisTauMass = cms.double(1.777),
    TrackSourceTag = cms.InputTag("ctfWithMaterialTracks"),
    RecoAnalysisMatchingDeltaR = cms.double(0.01),
    MuonSourceTag = cms.InputTag("muons"),
    RecoAnalysisMatchingPt = cms.double(0.1),
    RecoAnalysisTauMassCut = cms.double(0.2)
)


