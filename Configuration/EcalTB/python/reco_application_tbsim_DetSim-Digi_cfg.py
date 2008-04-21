import FWCore.ParameterSet.Config as cms

process = cms.Process("Sim")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("SimG4CMS.EcalTestBeam.test.crystal248_cff")

#
#Geometry
#   
process.load("Geometry.EcalTestBeam.TBH4GeometryXML_cfi")

#
# SIM : simulated hits + additional H4 TB objects 
#
process.load("Configuration.EcalTB.simulation_tbsim_cff")

#
# DIGI : H4 TB digitization 
#
process.load("Configuration.EcalTB.digitization_tbsim_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        SimEcalTBG4Object = cms.untracked.uint32(5432),
        ecalUnsuppressedDigis = cms.untracked.uint32(54321),
        VtxSmeared = cms.untracked.uint32(12345)
    ),
    sourceSeed = cms.untracked.uint32(98765)
)

process.source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        process.common_beam_direction_parameters,
        MaxE = cms.untracked.double(120.0),
        MinE = cms.untracked.double(120.0),
        # you can request more than 1 particle
        PartID = cms.untracked.vint32(11)
    ),
    Verbosity = cms.untracked.int32(0) ## set to 1 (or greater)  for printouts

)

process.VtxSmeared = cms.EDFilter("BeamProfileVtxGenerator",
    process.common_beam_direction_parameters,
    BeamSigmaX = cms.untracked.double(2.4),
    BeamSigmaY = cms.untracked.double(2.4),
    GaussianProfile = cms.untracked.bool(False)
)

process.o1 = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *', 
        'drop PSimHits_g4SimHits_*_Sim', 
        'keep PCaloHits_g4SimHits_EcalHitsEB_Sim', 
        'keep PCaloHits_g4SimHits_CaloHitsTk_Sim', 
        'keep PCaloHits_g4SimHits_EcalTBH4BeamHits_Sim'),
    fileName = cms.untracked.string('ECALH4TB_detsim_digi.root')
)

process.doSimHits = cms.Sequence(process.VtxSmeared*process.g4SimHits)
process.doSimTB = cms.Sequence(process.SimEcalTBG4Object*process.SimEcalTBHodoscope*process.SimEcalEventHeader)
process.doEcalDigis = cms.Sequence(process.mix*process.ecalUnsuppressedDigis)
process.p1 = cms.Path(process.doSimHits*process.doSimTB*process.doEcalDigis)
process.outpath = cms.EndPath(process.o1)
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    type = cms.string('EcalTBH4Trigger'),
    verbose = cms.untracked.bool(False),
    #IMPORTANT    #    #    #    #    #    #    #    # NUMBER OF EVENTS TO BE TRIGGERED 
    trigEvents = cms.untracked.int32(5)
))
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_EMV'

