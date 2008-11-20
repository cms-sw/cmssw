import FWCore.ParameterSet.Config as cms

dtResolutionAnalysisTest = cms.EDAnalyzer("DTResolutionAnalysisTest",
    folderRoot = cms.untracked.string(''),
    diagnosticPrescale = cms.untracked.int32(1),
    permittedMeanRange = cms.untracked.double(0.0005),
    permittedSigmaRange = cms.untracked.double(0.001)
 )


