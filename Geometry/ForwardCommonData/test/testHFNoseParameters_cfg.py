import FWCore.ParameterSet.Config as cms

process = cms.Process("HFnoseParametersTest")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load("Geometry.ForwardCommonData.testHFNoseXML_cfi")
process.load("Geometry.ForwardCommonData.hfnoseParametersInitialization_cfi")
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
        MinEta = cms.double(3.5),
        MaxEta = cms.double(5.5),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinE   = cms.double(9.99),
        MaxE   = cms.double(10.01)
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity       = cms.untracked.int32(0),
    firstRun        = cms.untracked.uint32(1)
)
 
process.testNose = cms.EDAnalyzer("HGCalParameterTester",
                                  Name = cms.string("HGCalHFNoseSensitive"),
                                  Mode = cms.int32(1)
)
 
process.p1 = cms.Path(process.generator*process.testNose)
