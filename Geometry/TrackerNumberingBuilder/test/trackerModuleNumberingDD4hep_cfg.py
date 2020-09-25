import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep
process = cms.Process('NumberingTest',Run3_dd4hep)

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load('Configuration.Geometry.GeometryDD4hepExtended2021Reco_cff')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.categories.append('TrackerGeometryBuilder')
    
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)


process.MessageLogger = cms.Service(
    "MessageLogger",
    statistics = cms.untracked.vstring('cout', 'tkmodulenumbering'),
    categories = cms.untracked.vstring('Geometry', 'ModuleNumbering'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
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

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.prod)


