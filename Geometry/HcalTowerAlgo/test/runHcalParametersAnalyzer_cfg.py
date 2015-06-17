import FWCore.ParameterSet.Config as cms
process = cms.Process("HcalParametersTest")

process.load("Configuration.Geometry.GeometryDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run1_mc']
process.GlobalTag.toGet = cms.VPSet(cms.PSet(record = cms.string('PHcalParametersRcd'),
                                             tag = cms.string('HCALParameters_Geometry_Run1_75YV2'),
                                             connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
                                             )
                                    )

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.hpa = cms.EDAnalyzer("HcalParametersAnalyzer")

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.hpa)
