import FWCore.ParameterSet.Config as cms

dtResolutionAnalysisTest = cms.EDAnalyzer("DTResolutionAnalysisTest",
    folderRoot = cms.untracked.string(''),
    sigmaTestName = cms.untracked.string('ResidualsSigmaInRange'),
    meanTestName = cms.untracked.string('ResidualsMeanInRange'),
    diagnosticPrescale = cms.untracked.int32(1)
 )


