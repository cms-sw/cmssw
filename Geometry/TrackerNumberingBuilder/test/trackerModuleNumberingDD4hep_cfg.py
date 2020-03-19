import FWCore.ParameterSet.Config as cms

process = cms.Process("NumberingTest")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.DD4hep_GeometrySim_cff")
process.load("Geometry.TrackerNumberingBuilder.DD4hep_trackerNumberingGeometry_cfi")

#this is always needed if users want access to the vector<GeometricDetExtra>
process.load("Geometry.TrackerNumberingBuilder.DD4hep_trackerNumberingExtraGeometry_cfi")

process.load("Geometry.TrackerGeometryBuilder.DD4hep_trackerParameters_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerTopology_cfi")
process.load("Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff")
process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.MessageLogger = cms.Service(
    "MessageLogger",
    statistics = cms.untracked.vstring('cout', 'tkmodulenumbering'),
    categories = cms.untracked.vstring('Geometry', 'ModuleNumbering'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        noLineBreaks = cms.untracked.bool(True)
        ),
    tkmodulenumbering = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            ),
        noLineBreaks = cms.untracked.bool(True),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            ),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            ),
        ERROR = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            ),
        threshold = cms.untracked.string('INFO'),
        Geometry = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
            ),
        ModuleNumbering = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
            )
        ),
    destinations = cms.untracked.vstring('cout',
                                         'tkmodulenumbering')
    )

process.prod = cms.EDAnalyzer("ModuleNumbering")
process.test = cms.EDAnalyzer("DDTestVectors",
                              DDDetector = cms.ESInputTag('','')
)
process.p1 = cms.Path(process.prod)


