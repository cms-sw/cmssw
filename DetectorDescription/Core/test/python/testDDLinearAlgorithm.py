import FWCore.ParameterSet.Config as cms

process = cms.Process("DDLinearTest")
process.load("DetectorDescription.Core.test.DDLinearXML_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.test = cms.EDAnalyzer("PerfectGeometryAnalyzer",
                              dumpPosInfo = cms.untracked.bool(True),
                              label = cms.untracked.string(''),
                              isMagField = cms.untracked.bool(False),
                              dumpSpecs = cms.untracked.bool(False),
                              dumpGeoHistory = cms.untracked.bool(True),
                              outFileName = cms.untracked.string('DDLinear.out'),
                              numNodesToDump = cms.untracked.uint32(0),
                              fromDB = cms.untracked.bool(False),
                              ddRootNodeName = cms.untracked.string('world:MotherOfAllBoxes')
                              )

process.Timing = cms.Service("Timing")

process.p = cms.Path(process.test)
