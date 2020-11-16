import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep

process = cms.Process("TrackerParametersTest", Run3_dd4hep)
process.load('Configuration.Geometry.GeometryDD4hepExtended2021Reco_cff')

if 'MessageLogger' in process.__dict__:
     process.MessageLogger.categories.append('TrackerParametersAnalyzer')
     process.MessageLogger.destinations.append('cout')

process.MessageLogger = cms.Service(
    "MessageLogger",
    statistics = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('TrackerParametersAnalyzer'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
            ),
        TrackerParametersAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
            ),
        noLineBreaks = cms.untracked.bool(True)
        ),
    trackerParametersDD4hep = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            ),
        FWKINFO = cms.untracked.PSet(
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
        TrackerParametersAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
            ),
        ),
    destinations = cms.untracked.vstring('cout',
                                         'trackerParametersDD4hep')
)
     
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.test = cms.EDAnalyzer("TrackerParametersAnalyzer")

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")
 
process.p1 = cms.Path(process.test)
