import FWCore.ParameterSet.Config as cms

process = cms.Process("NumberingTest")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.Geometry.GeometryReco_cff")
process.load("Geometry.CMSCommonData.cmsExtendedGeometryXML_cfi")
process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")

#this is always needed if users want access to the vector<GeometricDetExtra>
process.TrackerGeometricDetExtraESModule = cms.ESProducer( "TrackerGeometricDetExtraESModule",
                                                            fromDDD = cms.bool( True )
                                                            )
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.MessageLogger = cms.Service(
    "MessageLogger",
    statistics = cms.untracked.vstring('cout', 'tkModuleNumbering'),
    categories = cms.untracked.vstring('Geometry', 'ModuleNumbering'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        noLineBreaks = cms.untracked.bool(True)
        ),
    tkModuleNumbering = cms.untracked.PSet(
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
                                         'tkModuleNumbering')
    )

process.prod = cms.EDAnalyzer("ModuleNumbering")

process.p1 = cms.Path(process.prod)


