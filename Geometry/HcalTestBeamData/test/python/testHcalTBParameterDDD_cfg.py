import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load("SimG4CMS.HcalTestBeam.TB2004GeometryXML_cfi")
#process.load("SimG4CMS.HcalTestBeam.TB2002GeometryXML_cfi")
process.load("Geometry.EcalCommonData.ecalSimulationParameters_cff")
process.load("Geometry.HcalTestBeamData.hcalDDDSimConstants_cff")
process.load('GeneratorInterface.Core.generatorSmeared_cfi')
process.load("Configuration.EventContent.EventContent_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.HCalGeom=dict()
    process.MessageLogger.EcalGeom=dict()
    process.MessageLogger.HcalSim=dict()
    process.MessageLogger.HcalTBSim=dict()
    process.MessageLogger.EcalSim=dict()
    process.MessageLogger.CaloSim=dict()

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.common_beam_direction_parameters = cms.PSet(
    MinEta       = cms.double(0.5655),
    MaxEta       = cms.double(0.5655),
    MinPhi       = cms.double(-0.1309),
    MaxPhi       = cms.double(-0.1309),
    BeamPosition = cms.double(-521.5)
)

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
process.VtxSmeared = cms.EDProducer("BeamProfileVtxGenerator",
    process.common_beam_direction_parameters,
    VtxSmearedCommon,
    BeamMeanX       = cms.double(0.0),
    BeamMeanY       = cms.double(0.0),
    BeamSigmaX      = cms.double(0.0001),
    BeamSigmaY      = cms.double(0.0001),
    Psi             = cms.double(999.9),
    GaussianProfile = cms.bool(False),
    BinX            = cms.int32(50),
    BinY            = cms.int32(50),
    File            = cms.string('beam.profile'),
    UseFile         = cms.bool(False),
    TimeOffset      = cms.double(0.)
)

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        process.common_beam_direction_parameters,
        MinE   = cms.double(9.99),
        MaxE   = cms.double(10.01),
        PartID = cms.vint32(211)
    ),
    Verbosity       = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(False),
    firstRun        = cms.untracked.uint32(1)
)

process.Timing = cms.Service("Timing")

process.testH4 = cms.EDAnalyzer("HcalTBParameterTester",
                                Name = cms.untracked.string("EcalHitsEB"),
                                Mode = cms.untracked.int32(1)
)

process.testH2EE = cms.EDAnalyzer("HcalTBParameterTester",
                                  Name = cms.untracked.string("EcalHitsEB"),
                                  Mode = cms.untracked.int32(0)
)

process.testH2HC = cms.EDAnalyzer("HcalTBParameterTester",
                                  Name = cms.untracked.string("HcalHits"),
                                  Mode = cms.untracked.int32(0)
)

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.generatorSmeared*process.testH4)
#process.p1 = cms.Path(process.generator*process.VtxSmeared*process.generatorSmeared*process.testH2EE*process.testH2HC)
