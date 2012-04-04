import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.source = cms.Source("EmptySource",
)

process.demo = cms.EDAnalyzer("MapTester",
#    mapIOV = cms.uint32(1), # HO pre 2009
#    mapIOV = cms.uint32(2), # HO 2009 until May 6
#    mapIOV = cms.uint32(3), # HO May 6 2009 and onwards
    mapIOV = cms.uint32(4), # ZDC Oct 12 2009 and onwards
    generateTextfiles = cms.bool(True),
    generateEmap      = cms.bool(True),
)

process.p = cms.Path(process.demo)
