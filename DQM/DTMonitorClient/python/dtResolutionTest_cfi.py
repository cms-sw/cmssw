import FWCore.ParameterSet.Config as cms

resolutionTest = cms.EDFilter("DTResolutionTest",
    runningStandalone = cms.untracked.bool(True),
    folderRoot = cms.untracked.string(''),
    sigmaTestName = cms.untracked.string('ResidualsSigmaInRange'),
    meanTestName = cms.untracked.string('ResidualsMeanInRange'),
    #Names of the quality tests: they must match those specified in "qtList"
    resDistributionTestName = cms.untracked.string('ResidualsDistributionGaussianTest'),
    debug = cms.untracked.bool(False),
    diagnosticPrescale = cms.untracked.int32(1),
    histoTag = cms.untracked.string('hResDist')
)


