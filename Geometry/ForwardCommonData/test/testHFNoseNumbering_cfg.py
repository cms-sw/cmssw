import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load("Geometry.ForwardCommonData.testHFNoseXML_cfi")
process.load("Geometry.ForwardCommonData.hfnoseParametersInitialization_cfi")
process.load("Geometry.ForwardCommonData.hfnoseNumberingInitialization_cfi")
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

process.prodNose = cms.EDAnalyzer("HGCalNumberingTester",
    NameSense     = cms.string("HGCalHFNoseSensitive"),
    NameDevice    = cms.string("HFNose"),
    LocalPositionX= cms.vdouble(50.0,80.0,110.0,130.0),
    LocalPositionY= cms.vdouble(50.0,0.0,0.0,0.0),
    Increment     = cms.int32(2),	
    DetType       = cms.int32(2),
    Reco          = cms.bool(False)
)
 
process.p1 = cms.Path(process.generator*process.prodNose)

