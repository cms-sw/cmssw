import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C11_cff import Phase2C11

process = cms.Process("PROD",Phase2C11)

process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load("Geometry.HGCalCommonData.testHGCalV15XML_cfi")
process.load("Geometry.HGCalCommonData.hgcalV15ParametersInitialization_cfi")
process.load("Geometry.HGCalCommonData.hgcalNumberingInitialization_cfi")
process.load("Geometry.EcalCommonData.ecalSimulationParameters_cff")
process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalGeom=dict()

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
                                   PGunParameters = cms.PSet(
                                       PartID = cms.vint32(14),
                                       MinEta = cms.double(-3.5),
                                       MaxEta = cms.double(3.5),
                                       MinPhi = cms.double(-3.14159265359),
                                       MaxPhi = cms.double(3.14159265359),
                                       MinE   = cms.double(9.99),
                                       MaxE   = cms.double(10.01)
                                   ),
                                   AddAntiParticle = cms.bool(False),
                                   Verbosity       = cms.untracked.int32(0),
                                   firstRun        = cms.untracked.uint32(1)
                               )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("Geometry.HGCalCommonData.hgcalNumberingTesterEE_cfi")
process.hgcalNumberingTesterEE.LocalPositionX= [500.0,350.0,800.0,1400.0]
process.hgcalNumberingTesterEE.LocalPositionY= [500.0,0.0,0.0,0.0]

process.hgcalNumberingTesterHEF = process.hgcalNumberingTesterEE.clone(
    NameSense  = "HGCalHESiliconSensitive",
    NameDevice = "HGCal HE Front",
    Increment  = 9
)
 
process.hgcalNumberingTesterHEB = process.hgcalNumberingTesterEE.clone(
    NameSense  = "HGCalHEScintillatorSensitive",
    NameDevice = "HGCal HE Back",
    Increment  = 9,
    LocalPositionX= [1100.0,1400.0,1500.0,1600.0],
    LocalPositionY= [1100.0,1000.0,500.0,0.0],
    DetType    = 0
)
 
process.p1 = cms.Path(process.generator*process.hgcalNumberingTesterEE*process.hgcalNumberingTesterHEF*process.hgcalNumberingTesterHEB)
