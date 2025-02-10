import FWCore.ParameterSet.Config as cms
from Configuration.AlCa.autoCond import autoCond

process = cms.Process("GeometryTest")

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(3),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi')
process.GlobalTag.globaltag = autoCond['run1_mc']

process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(record = cms.string("PHGCalRcd"),
             tag = cms.string("HGCALRECO_Geometry_Test01"),
             connect = cms.string("sqlite_file:myfile.db")
             )
    )

process.MessageLogger = cms.Service("MessageLogger")
process.demo = cms.EDAnalyzer("PrintEventSetupContent")

process.GeometryTester = cms.EDAnalyzer("GeometryTester",
                                        XMLTest = cms.untracked.bool(False),
                                        TrackerTest = cms.untracked.bool(False),
                                        EcalTest = cms.untracked.bool(False),
                                        HcalTest = cms.untracked.bool(False),
                                        HGCalTest = cms.untracked.bool(True),
                                        CaloTowerTest = cms.untracked.bool(False),
                                        CastorTest = cms.untracked.bool(False),
                                        ZDCTest = cms.untracked.bool(False),
                                        CSCTest = cms.untracked.bool(False),
                                        DTTest = cms.untracked.bool(False),
                                        RPCTest = cms.untracked.bool(False),
                                        geomLabel = cms.untracked.string("")
                                        )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.GeometryTester) #replace this with process.demo if you want to see the PrintEventSetupContent output.

