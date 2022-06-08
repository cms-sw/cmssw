import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C11_dd4hep_cff import Phase2C11_dd4hep

process = cms.Process("HGCalParametersTest",Phase2C11_dd4hep)
process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load("Configuration.Geometry.GeometryDD4hepExtended2026D77Reco_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalGeom=dict()


process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

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

process.load("Geometry.HGCalCommonData.hgcParameterTesterEE_cfi")
#process.hgcParameterTesterEE.Mode = 0

process.hgcParameterTesterHESil = process.hgcParameterTesterEE.clone(
    Name = cms.string("HGCalHESiliconSensitive")
)

process.hgcParameterTesterHESci = process.hgcParameterTesterEE.clone(
    Name = cms.string("HGCalHEScintillatorSensitive"),
    Mode = cms.int32(2)
)
 
process.p1 = cms.Path(process.generator*process.hgcParameterTesterEE*process.hgcParameterTesterHESil*process.hgcParameterTesterHESci)
#process.p1 = cms.Path(process.generator*process.hgcParameterTesterEE*process.hgcParameterTesterHESil)
