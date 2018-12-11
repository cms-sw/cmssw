import FWCore.ParameterSet.Config as cms

process = cms.Process("DDCMSDetectorTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.string('DetectorDescription/DDCMS/data/cms-2015-muon-geometry.xml')
                                            )

process.test = cms.EDAnalyzer("DDTestNavigateGeometry")

process.p = cms.Path(process.test)
