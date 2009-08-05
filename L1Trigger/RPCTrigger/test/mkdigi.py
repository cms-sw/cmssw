import FWCore.ParameterSet.Config as cms

# This gives the name to the produced trees in the rootfile
process = cms.Process("SIMDIGI")

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
     destinations = cms.untracked.vstring('digi')
)
                                    

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)


process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        VtxSmeared = cms.untracked.uint32(123456789),
        simMuonRPCDigis = cms.untracked.uint32(563),
        mix = cms.untracked.uint32(9823),
        generator = cms.untracked.uint32(135744645)
    )
)

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        MinPt = cms.double(2),
        MaxPt = cms.double(80),
        PartID = cms.vint32(13,-13),
        MaxEta = cms.double(0.1),
        MaxPhi = cms.double(3.141592),
        MinEta = cms.double(-0.1),
        MinPhi = cms.double(-3.141592)
    ),
    Verbosity = cms.untracked.int32(0),
    firstRun = cms.untracked.uint32(100),
    AddAntiParticle = cms.bool(False) 
)


process.out = cms.OutputModule("PoolOutputModule",
    # use process below if you want to keep the digis in case of you don't emulate the trigger 
    # outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    fileName = cms.untracked.string('digi.root'),

    outputCommands = cms.untracked.vstring(
                        "drop *",
                        "keep RPCDetIdRPCDigiMuonDigiCollection_*_*_*",
                        "keep SimTracks_*_*_*",
                        "keep *_*_MuonCSCHits_*",
                        "keep *_*_MuonDTHits_*",
                        "keep *_*_MuonRPCHits_*",
                        "keep CrossingFramePlaybackInfo_*_*_*"
     )


)

# regular things
process.GenSimDigi = cms.Sequence(process.generator*process.VtxSmeared*process.g4SimHits*process.mix*process.simMuonRPCDigis)

process.p = cms.Path(process.GenSimDigi)
process.this_is_the_end = cms.EndPath(process.out)



