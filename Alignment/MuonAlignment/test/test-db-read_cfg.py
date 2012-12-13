import FWCore.ParameterSet.Config as cms

sqlite_file = "initialMuonAlignment_DT-Aug11_CSC-Aug12.xml.phizsl13_5mrad3.db"

process = cms.Process("TEST")

# Message logger service
process.load("FWCore.MessageService.MessageLogger_cfi")

# DT and CSC geometry 
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
#process.load("Geometry.DTGeometry.dtGeometry_cfi")
#process.load("Geometry.CSCGeometry.cscGeometry_cfi")

### (don't load dtGeometry_cfi or cscGeometry_cfi because it's provided by AlignmentProducer)
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("RecoTracker.Configuration.RecoTracker_cff")
#del process.DTGeometryESModule
#del process.CSCGeometryESModule


process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string("GR_R_44_V7::All")

from CondCore.DBCommon.CondDBSetup_cfi import *

process.MuAlInputDB = cms.ESSource("PoolDBESSource",
                           CondDBSetup,
                           connect = cms.string("sqlite_file:%s" % sqlite_file),
                           toGet = cms.VPSet(cms.PSet(record = cms.string("DTAlignmentRcd"), tag = cms.string("DTAlignmentRcd")),
                                             cms.PSet(record = cms.string("DTAlignmentErrorRcd"), tag = cms.string("DTAlignmentErrorRcd")),
                                             cms.PSet(record = cms.string("CSCAlignmentRcd"), tag = cms.string("CSCAlignmentRcd")),
                                             cms.PSet(record = cms.string("CSCAlignmentErrorRcd"), tag = cms.string("CSCAlignmentErrorRcd"))))
process.es_prefer_MuAlInputDB = cms.ESPrefer("PoolDBESSource", "MuAlInputDB")


process.GPRInputDB = cms.ESSource("PoolDBESSource",
                                  CondDBSetup,
                                  connect = cms.string("frontier://FrontierProd/CMS_COND_31X_ALIGNMENT"),
                                  toGet = cms.VPSet(cms.PSet(record = cms.string("GlobalPositionRcd"), tag = cms.string("GlobalAlignment_v4_offline"))))
process.es_prefer_GlobalPositionInputDB = cms.ESPrefer("PoolDBESSource", "GPRInputDB")


process.prod = cms.EDAnalyzer("TestMuonReader")

process.p1 = cms.Path(process.prod)
