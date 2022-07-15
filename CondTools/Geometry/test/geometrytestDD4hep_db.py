import FWCore.ParameterSet.Config as cms
from Configuration.AlCa.autoCond import autoCond

from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep

process = cms.Process("GeometryWriter", Run3_dd4hep)

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

process.load('CondCore.CondDB.CondDB_cfi')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = autoCond['updgrade2021']

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
                                        HGCalTest = cms.untracked.bool(False),
                                        CaloTowerTest = cms.untracked.bool(True),
                                        CastorTest = cms.untracked.bool(False),
                                        ZDCTest = cms.untracked.bool(True),
                                        CSCTest = cms.untracked.bool(True),
                                        DTTest = cms.untracked.bool(True),
                                        RPCTest = cms.untracked.bool(True),
                                        geomLabel = cms.untracked.string("Extended"),
                                        roundValues = cms.untracked.bool(False)
                                        )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.GeometryTester)
