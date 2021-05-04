# Run CSCIndexerAnalyzer2 - ptc - 20.11.2012

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
process.source = cms.Source("EmptySource")

process.load("Configuration.Geometry.GeometryExtended_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Geometry.CommonTopologies.globalTrackingGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.GlobalTag.globaltag = "MC_61_V2::All"
process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource")

process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.dummy = cms.ESSource("EmptyESSource",
    recordName = cms.string("CSCIndexerRecord"),
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True)
)

##process.load("CalibMuon.CSCCalibration.CSCIndexer_cfi")
## The above cfi gives rise to the following line:

##process.CSCIndexerESProducer = cms.ESProducer("CSCIndexerESProducer", AlgoName = cms.string("CSCIndexerStartup") )
process.CSCIndexerESProducer = cms.ESProducer("CSCIndexerESProducer", AlgoName = cms.string("CSCIndexerPostls1") )

process.analyze = cms.EDAnalyzer("CSCIndexerAnalyzer2")

process.test = cms.Path(process.analyze)

