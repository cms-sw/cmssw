import FWCore.ParameterSet.Config as cms

# This gives the name to the produced trees in the rootfile
process = cms.Process("SIMDIGI")

process.load("SimGeneral.HepPDTESSource.pdt_cfi")

# 1 Generate (process: source)
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# 2 Vtx Smearing
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

# 3 Mix for (no) Pile Up
process.load("Configuration.StandardSequences.MixingNoPileUp_cff")

# 4 Simulate Hits
process.load("Configuration.StandardSequences.Sim_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load('Configuration.StandardSequences.GeometryExtended_cff')

#process.load("Configuration.StandardSequences.GeometryNoCastor_cff")

# 5 Digitize SimHits
#process.load("CalibMuon.RPCCalibration.RPCFakeESProducer_cfi")
#process.load("CalibMuon.RPCCalibration.RPC_Calibration_cff")
process.load("SimMuon.RPCDigitizer.muonRPCDigis_cfi")

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = 'MC_37Y_V1::All'


process.MessageLogger = cms.Service("MessageLogger",
     destinations = cms.untracked.vstring('digi')
)
                                    


process.MessageLogger = cms.Service("MessageLogger",
     destinations = cms.untracked.vstring('digi')
)
                                    

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
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
        MinPt = cms.double(90),
        MaxPt = cms.double(99),
        PartID = cms.vint32(13,-13,13,-13,13,-13),
        MaxEta = cms.double(1.6),
#MaxPhi = cms.double(3.141592),
        MaxPhi = cms.double(-0.21),
        MinEta = cms.double(-1.6),
#       MinPhi = cms.double(-3.141592)
        MinPhi = cms.double(-0.22)
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



