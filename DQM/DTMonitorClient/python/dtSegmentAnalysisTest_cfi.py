import FWCore.ParameterSet.Config as cms

segmentTest = cms.EDFilter("DTSegmentAnalysisTest",
    #Permetted value of chi2 segment quality
    chi2Threshold = cms.untracked.double(5.0),
    #Permetted percentual of CH with bad chi2 segments 
    badCHpercentual = cms.untracked.int32(10),
    #Permetted percentual of segments with bad chi2
    badSegmPercentual = cms.untracked.int32(30),
    diagnosticPrescale = cms.untracked.int32(1),
    folderRoot = cms.untracked.string('')
)


