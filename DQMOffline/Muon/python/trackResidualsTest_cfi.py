import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

muTrackResidualsTest = DQMEDHarvester("MuonTrackResidualsTest",
    sigmaTestName = cms.untracked.string('ResidualsSigmaInRange'),
    meanTestName = cms.untracked.string('ResidualsMeanInRange'),
    # number of luminosity block to analyse
    diagnosticPrescale = cms.untracked.int32(1),
    # quality test name
    resDistributionTestName = cms.untracked.string('ResidualsDistributionGaussianTest')
)



