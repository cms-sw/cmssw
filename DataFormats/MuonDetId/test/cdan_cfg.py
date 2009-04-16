# Configuration file to run CSCDetIdAnalyzer
# to dump CSC geometry focussing on CSCDetId
# Tim Cox 12.06.2008 pythonized

import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCIndexerTest")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonCommonData.muonEndcapIdealGeometryXML_cfi")

# flags for modelling of CSC layer & strip geometry
process.load("Geometry.CSCGeometry.cscGeometry_cfi")

# fake alignment nonsense. I wish I didn't need this junk when all I want is ideal geometry!
process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.fake2 = process.FakeAlignmentSource
del process.FakeAlignmentSource
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource", "fake2")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# Not sure the following does anything useful :(
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.debugModules.append('process.analyzer')
process.MessageLogger.categories.append('CSC')
process.MessageLogger.cout = cms.untracked.PSet(
  threshold = cms.untracked.string('DEBUG'),
  default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
  FwkReport = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
  CSC       = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)

process.analyzer = cms.EDAnalyzer(
    "CSCDetIdAnalyzer",
    ## For unganged ME1a (as for SLHC) set next param to True
    UngangedME1a=cms.untracked.bool(False)
)

process.p1 = cms.Path(process.analyzer)

