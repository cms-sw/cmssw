import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.source = cms.Source("EmptySource",
)

process.demo = cms.EDAnalyzer("MapTester",
#    mapIOV = cms.uint32(1), # pre 2009
    mapIOV = cms.uint32(2), # 2009 until May 6
#    mapIOV = cms.uint32(3), # May 6 2009 and onwards
    generateTextfiles = cms.bool(False),
    generateEmap      = cms.bool(False),

)

process.p = cms.Path(process.demo)
