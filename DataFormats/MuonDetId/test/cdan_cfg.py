# Configuration file to run CSCDetIdAnalyzer
# to dump CSC geometry focussing on CSCDetId
# Tim Cox 12.06.2008 pythonized

import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonCommonData.muonEndcapIdealGeometryXML_cfi")
# flags for modelling of CSC layer & strip geometry
process.load("Geometry.CSCGeometry.cscGeometry_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.MessageLogger = cms.Service("MessageLogger",
    # For LogDebug/LogTrace output...
    #    untracked vstring debugModules   = { "*" }
    # No constraint on log.out content...equivalent to threshold INFO
    # 0 means none, -1 means all (?)
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        CSC = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        noLineBreaks = cms.untracked.bool(True)
    ),
    categories = cms.untracked.vstring('CSC'),
    destinations = cms.untracked.vstring('cout')
)

process.producer = cms.EDAnalyzer("CSCDetIdAnalyzer")

process.p1 = cms.Path(process.producer)

