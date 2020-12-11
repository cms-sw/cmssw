import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep

process = cms.Process("TrackerParametersTest", Run3_dd4hep)
process.load('Configuration.Geometry.GeometryDD4hepExtended2021Reco_cff')

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
        trackerParametersDD4hep = cms.untracked.PSet(
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
     
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.test = cms.EDAnalyzer("TrackerParametersAnalyzer")

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")
 
process.p1 = cms.Path(process.test)
