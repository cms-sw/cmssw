import FWCore.ParameterSet.Config as cms

# MuonAlignmentAnalyzer
muonAlignmentSummary = cms.EDAnalyzer("MuonAlignmentSummary",
    doDT = cms.untracked.bool(True),
    doCSC = cms.untracked.bool(True),
    meanPositionRange = cms.untracked.double(0.5),
    rmsPositionRange = cms.untracked.double(0.1),
    meanAngleRange = cms.untracked.double(0.1),
    rmsAngleRange = cms.untracked.double(0.01),
 )



