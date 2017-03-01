import FWCore.ParameterSet.Config as cms
process = cms.Process("HcalParametersTest")

process.load("Configuration.Geometry.GeometryDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run1_mc']
process.GlobalTag.snapshotTime = cms.string("9999-12-31 23:59:59.000")
process.GlobalTag.toGet = cms.VPSet(cms.PSet(record = cms.string('HcalParametersRcd'),
                                             tag = cms.string('HCALParameters_Geometry_Test01'),
                                             connect = cms.string("sqlite_file:myfile.db")
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
