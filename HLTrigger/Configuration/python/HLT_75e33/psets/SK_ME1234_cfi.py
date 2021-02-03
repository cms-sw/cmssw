import FWCore.ParameterSet.Config as cms

SK_ME1234 = cms.PSet(
    chi2Max = cms.double(99999.0),
    dPhiFineMax = cms.double(0.025),
    dPhiMax = cms.double(0.003),
    dRPhiFineMax = cms.double(8.0),
    dRPhiMax = cms.double(8.0),
    minLayersApart = cms.int32(2),
    verboseInfo = cms.untracked.bool(True),
    wideSeg = cms.double(3.0)
)