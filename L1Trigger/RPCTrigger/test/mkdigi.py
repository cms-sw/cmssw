import FWCore.ParameterSet.Config as cms

process = cms.Process("genDigi")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

process.load("Geometry.CMSCommonData.cmsSimIdealGeometryXML_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("SimMuon.RPCDigitizer.muonRPCDigis_cfi")

process.load("CalibMuon.RPCCalibration.RPCFakeESProducer_cfi")

process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('detailedInfo.txt')
)

process.Timing = cms.Service("Timing")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        simMuonRPCDigis = cms.untracked.uint32(21),
        g4SimHits = cms.untracked.uint32(333),
        VtxSmeared = cms.untracked.uint32(123456789)
    ),
    sourceSeed = cms.untracked.uint32(135799753)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("FlatRandomPtGunSource",
    PGunParameters = cms.untracked.PSet(
        MaxPt = cms.untracked.double(140.0),
        MinPt = cms.untracked.double(50.0),
        PartID = cms.untracked.vint32(-13),
        MaxEta = cms.untracked.double(1.4),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-1.4),
        MinPhi = cms.untracked.double(-3.14159265359)
    ),
    Verbosity = cms.untracked.int32(0),
    firstRun = cms.untracked.uint32(2),
    AddAntiParticle = cms.untracked.bool(False) 

)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('digi0T.root')
)

process.p1 = cms.Path(process.VtxSmeared*process.g4SimHits*process.mix*process.simMuonRPCDigis)
process.outpath = cms.EndPath(process.out)


