import FWCore.ParameterSet.Config as cms

process = cms.Process("DDCMSDetectorTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.test = cms.EDAnalyzer("DDCMSDetector",
                              geomXMLFiles = cms.string('DetectorDescription/DDCMS/data/cms-tracker.xml')
                              )

process.p = cms.Path(process.test)
