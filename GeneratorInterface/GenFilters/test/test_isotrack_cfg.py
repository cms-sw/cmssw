import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.Generator.QCDForPF_14TeV_TuneCUETP8M1_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
                                                   generator = cms.PSet(
                                                       initialSeed = cms.untracked.uint32(123456789),
                                                       engineName = cms.untracked.string('HepJamesRandom')
    )
)

# The following three lines reduce the clutter of repeated printouts
# of the same exception message.
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.PythiaFilter=dict()

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))

process.source = cms.Source("EmptySource")

process.load("GeneratorInterface.GenFilters.PythiaFilterIsolatedTrack_cfi")

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('QCD14TeVIsoTrack.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p')
    ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN')
    )

)

process.isotrack_filter.minSeedEta = 2.0
process.isotrack_filter.maxSeedEta = 3.0

process.p = cms.Path(process.generator * process.isotrack_filter)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)
