import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

# MuonAlignmentAnalyzer
muonAlignmentSummary = DQMEDHarvester("MuonAlignmentSummary",
    doDT = cms.untracked.bool(True),
    doCSC = cms.untracked.bool(True),
    meanPositionRange = cms.untracked.double(0.5),
    rmsPositionRange = cms.untracked.double(0.01),
    meanAngleRange = cms.untracked.double(0.05),
    rmsAngleRange = cms.untracked.double(0.005),
    FolderName = cms.string("Test/")
 )



