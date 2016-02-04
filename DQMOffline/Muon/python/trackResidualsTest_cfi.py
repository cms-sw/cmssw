import FWCore.ParameterSet.Config as cms

muTrackResidualsTest = cms.EDAnalyzer("MuonTrackResidualsTest",
    sigmaTestName = cms.untracked.string('ResidualsSigmaInRange'),
    meanTestName = cms.untracked.string('ResidualsMeanInRange'),
    # number of luminosity block to analyse
    diagnosticPrescale = cms.untracked.int32(1),
    # quality test name
    resDistributionTestName = cms.untracked.string('ResidualsDistributionGaussianTest')
)



