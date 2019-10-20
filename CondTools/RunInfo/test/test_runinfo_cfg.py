import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.maxEvents.input = 3

process.source = cms.Source("EmptySource",
                            firstLuminosityBlockForEachRun = cms.untracked.VLuminosityBlockID(
                                cms.LuminosityBlockID(10,1),
                                cms.LuminosityBlockID(20,2),
                                cms.LuminosityBlockID(30,3)
                            ),
                            numberEventsInLuminosityBlock =cms.untracked.uint32(1)
)

process.add_( cms.ESProducer("RunInfoTestESProducer",
                             runInfos = cms.VPSet(cms.PSet(run = cms.int32(10), avg_current = cms.double(1.)),
                                              cms.PSet(run = cms.int32(20), avg_current = cms.double(2.)),
                                              cms.PSet(run = cms.int32(30), avg_current = cms.double(3.)) ) ) )

process.riSource = cms.ESSource("EmptyESSource", recordName = cms.string("RunInfoRcd"),
                                iovIsRunNotTime = cms.bool(True),
                                firstValid = cms.vuint32(10,20,30))

process.test = cms.EDAnalyzer("RunInfoESAnalyzer")

process.p = cms.Path(process.test)
