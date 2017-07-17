import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
# Message logger service
process.load("FWCore.MessageService.MessageLogger_cfi")

# Ideal geometry producer
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

# testing ideal geometry from Fake...
#process.load("Geometry.DTGeometry.dtGeometry_cfi")
#process.load("Geometry.CSCGeometry.cscGeometry_cfi")
#process.load("CalibMuon.Configuration.Muon_FakeAlignment_cff")

##  ...or testing geometry from DB/sqlite...
#process.load("Geometry.DTGeometry.dtGeometry_cfi")
#process.load("Geometry.CSCGeometry.cscGeometry_cfi")
#process.load("Alignment.CommonAlignmentProducer.GlobalPosition_Frontier_cff")
#
#from CondCore.DBCommon.CondDBSetup_cfi import *
#process.PoolDBESSource = cms.ESSource(
#    "PoolDBESSource",
#    CondDBSetup,
#    #connect = cms.string('frontier://FrontierProd/CMS_COND_31X_ALIGNMENT'),
#    connect = cms.string('sqlite_file:Alignments.db'),
#    toGet = cms.VPSet(cms.PSet(record = cms.string('DTAlignmentRcd'),
#                               tag = cms.string('DT100InversepbScenario')
#                               ), 
#                      cms.PSet(record = cms.string('DTAlignmentErrorExtendedRcd'),
#                               tag = cms.string('DT100InversepbScenarioErrors')
#                               ), 
#                      cms.PSet(record = cms.string('CSCAlignmentRcd'),
#                               tag = cms.string('CSC100InversepbScenario')
#                               ), 
#                      cms.PSet(record = cms.string('CSCAlignmentErrorExtendedRcd'),
#                               tag = cms.string('CSC100InversepbScenarioErrors')
#                               )
#                      )
#    )

## ... or testing any of the scenarios from cff
process.load("Alignment.MuonAlignment.MisalignedMuon_cfi")
import Alignment.MuonAlignment.Scenarios_cff as _Scenarios
#process.MisalignedMuon.scenario = _Scenarios.MuonNoMovementsScenario
#process.MisalignedMuon.scenario = _Scenarios.Muon10InversepbScenario
process.MisalignedMuon.scenario = _Scenarios.Muon10inversePbScenario2008

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.prod = cms.EDAnalyzer("TestMuonHierarchy")

process.p1 = cms.Path(process.prod)
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    )
)


