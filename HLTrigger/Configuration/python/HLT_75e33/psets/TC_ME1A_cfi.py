import FWCore.ParameterSet.Config as cms

TC_ME1A = cms.PSet(
    SegmentSorting = cms.int32(1),
    chi2Max = cms.double(6000.0),
    chi2ndfProbMin = cms.double(0.0001),
    dPhiFineMax = cms.double(0.013),
    dPhiMax = cms.double(0.00198),
    dRPhiFineMax = cms.double(3.0),
    dRPhiMax = cms.double(0.6),
    minLayersApart = cms.int32(2),
    verboseInfo = cms.untracked.bool(True)
)