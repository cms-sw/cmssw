import FWCore.ParameterSet.Config as cms

process = cms.Process("DDFilteredViewTest")

##process.load('Configuration.Geometry.GeometryExtended2015_cff')
process.load("Configuration.Geometry.GeometryDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run1_mc']
process.GlobalTag.toGet = cms.VPSet(cms.PSet(record = cms.string('PHcalParametersRcd'),
                                             tag = cms.string('HCALParameters_Geometry_Run1_75YV2'),
                                             connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
                                             )
                                    )

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.CMSGeom=dict()
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.fva = cms.EDAnalyzer("DDFilteredViewAnalyzer",
                             attribute = cms.string("OnlyForHcalSimNumbering"),
                             value = cms.string("any"))

process.p1 = cms.Path(process.fva)

