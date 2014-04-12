import FWCore.ParameterSet.Config as cms

process = cms.Process("DigFP420Test")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("Configuration.StandardSequences.VtxSmearedFlat_cff")

process.load("Geometry.FP420CommonData.FP420GeometryXML_cfi")

process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load("SimRomanPot.SimFP420.FP420Digi_cfi")

process.load("RecoRomanPot.RecoFP420.FP420Cluster_cfi")

process.load("RecoRomanPot.RecoFP420.FP420Track_cfi")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(11),
        mix = cms.untracked.uint32(12345),
        VtxSmeared = cms.untracked.uint32(98765432)
    ),
    sourceSeed = cms.untracked.uint32(123456789)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        PartID = cms.untracked.vint32(2212),
        MaxEta = cms.untracked.double(9.9),
        MaxPhi = cms.untracked.double(3.227),
        MinEta = cms.untracked.double(8.7),
        MinE = cms.untracked.double(6930.0),
        MinPhi = cms.untracked.double(3.053),
        MaxE = cms.untracked.double(7000.0)
    ),
    Verbosity = cms.untracked.int32(0)
)

process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testDICLTRevent.root')
)

process.Timing = cms.Service("Timing")

process.Tracer = cms.Service("Tracer")

process.p1 = cms.Path(process.VtxSmeared*process.g4SimHits*process.mix*process.FP420Digi*process.FP420Cluster*process.FP420Track)
process.outpath = cms.EndPath(process.o1)
process.schedule = cms.Schedule(process.p1,process.outpath)

process.MessageLogger.cerr.default.limit = 10
process.FlatVtxSmearingParameters.MinX = -0.5
process.FlatVtxSmearingParameters.MaxX = -2.5
process.FlatVtxSmearingParameters.MinY = 0.0
process.FlatVtxSmearingParameters.MaxY = 0.0
process.FlatVtxSmearingParameters.MinZ = 41800.
process.FlatVtxSmearingParameters.MaxZ = 41800.
process.FP420Digi.ApplyTofCut = False
process.FP420Cluster.VerbosityLevel = 1
#rocess.FP420Track.VerbosityLevel = 1
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Generator.ApplyPhiCuts = True
process.g4SimHits.Generator.ApplyEtaCuts = False
process.g4SimHits.NonBeamEvent = True

