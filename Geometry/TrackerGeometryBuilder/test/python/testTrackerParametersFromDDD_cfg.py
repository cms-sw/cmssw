import FWCore.ParameterSet.Config as cms

process = cms.Process("TrackerParametersTest")
process.load('Configuration.Geometry.GeometryExtended2021_cff')
process.load('Configuration.Geometry.GeometryExtended2021Reco_cff')

process.trackerGeometry.applyAlignment = cms.bool(False)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        TrackerParametersAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        enable = cms.untracked.bool(True),
        enableStatistics = cms.untracked.bool(True),
        noLineBreaks = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    ),
    files = cms.untracked.PSet(
        trackerParametersDDD = cms.untracked.PSet(
            DEBUG = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            ERROR = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            FWKINFO = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            INFO = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            TrackerParametersAnalyzer = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            WARNING = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            noLineBreaks = cms.untracked.bool(True),
            threshold = cms.untracked.string('INFO')
        )
    )
)

process.test = cms.EDAnalyzer("TrackerParametersAnalyzer")

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.test)



