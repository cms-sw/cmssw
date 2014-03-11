import FWCore.ParameterSet.Config as cms

# AM-based pattern recognition default sequence
TTPatternsFromStub = cms.EDProducer("TrackFindingAMProducer",
   TTInputStubs       = cms.InputTag("TTStubsFromPixelDigis", "StubAccepted"),
   TTPatternName      = cms.string("AML1Patterns"),
   inputBankFile      = cms.string('/afs/cern.ch/work/s/sviret/testarea/PatternBanks/BE_5D/Eta7_Phi8/ss32_cov40/612_SLHC6_MUBANK_lowmidhig_sec37_ss32_cov40.pbk'),
   threshold          = cms.int32(5)
)

# Hough-based trackfit default sequence
TTTracksFromPattern = cms.EDProducer("TrackFitHoughProducer",
   TTInputStubs       = cms.InputTag("TTStubsFromPixelDigis", "StubAccepted"),
   TTInputPatterns    = cms.InputTag("MergePROutput", "AML1Patterns"),
   TTTrackName        = cms.string("AML1Tracks"),
)

# AM output merging sequence
MergePROutput = cms.EDProducer("AMOutputMerger",
   TTInputClusters     = cms.InputTag("TTStubsFromPixelDigis", "ClusterAccepted"),
   TTInputStubs        = cms.InputTag("TTStubsFromPixelDigis", "StubAccepted"),
   TTInputPatterns     = cms.VInputTag(cms.InputTag("TTPatternsFromStub", "AML1Patterns")),                               
   TTFiltClustersName  = cms.string("ClusInPattern"),
   TTFiltStubsName     = cms.string("StubInPattern"),
   TTPatternsName      = cms.string("AML1Patterns")                         
)

MergeFITOutput = cms.EDProducer("AMOutputMerger",
   TTInputClusters     = cms.InputTag("TTStubsFromPixelDigis", "ClusterAccepted"),
   TTInputStubs        = cms.InputTag("TTStubsFromPixelDigis", "StubAccepted"),
   TTInputPatterns     = cms.VInputTag(cms.InputTag("TTTracksFromPattern", "AML1Tracks")),                               
   TTFiltClustersName  = cms.string("ClusInTrack"),
   TTFiltStubsName     = cms.string("StubInTrack"),
   TTPatternsName      = cms.string("AML1Tracks")                         
)
