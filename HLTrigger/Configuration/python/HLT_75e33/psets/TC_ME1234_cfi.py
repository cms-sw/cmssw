import FWCore.ParameterSet.Config as cms

TC_ME1234 = cms.PSet(
    SegmentSorting = cms.int32(1),
    chi2Max = cms.double(6000.0),
    chi2ndfProbMin = cms.double(0.0001),
    dPhiFineMax = cms.double(0.02),
    dPhiMax = cms.double(0.003),
    dRPhiFineMax = cms.double(6.0),
    dRPhiMax = cms.double(1.2),
    minLayersApart = cms.int32(2),
    verboseInfo = cms.untracked.bool(True)
)