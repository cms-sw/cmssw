import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryWriter")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load('Configuration/StandardSequences/GeometryDB_cff')

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.GeometryTester = cms.EDAnalyzer("GeometryTester",
                                        XMLTest = cms.untracked.bool(True),
                                        TrackerTest = cms.untracked.bool(True),
                                        EcalTest = cms.untracked.bool(True),
                                        HcalTest = cms.untracked.bool(True),
                                        CaloTowerTest = cms.untracked.bool(True),
                                        CastorTest = cms.untracked.bool(True),
                                        ZDCTest = cms.untracked.bool(True),
                                        CSCTest = cms.untracked.bool(True),
                                        DTTest = cms.untracked.bool(True),
                                        RPCTest = cms.untracked.bool(True),
                                        geomLabel = cms.untracked.string("Extended")
                                        )
 
 
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.GeometryTester)

