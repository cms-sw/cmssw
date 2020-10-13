import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load("Configuration.Geometry.GeometryExtended2026D71Reco_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.categories.append('HGCalGeom')

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

process.prodEE = cms.EDAnalyzer("HGCalWaferCheck",
                                NameSense     = cms.string("HGCalEESensitive"),
                                NameDevice    = cms.string("HGCal EE"),
                                Reco          = cms.bool(True)
)

process.prodHEF = process.prodEE.clone(
    NameSense  = "HGCalHESiliconSensitive",
    NameDevice = "HGCal HE Front",
)
 
 
process.p1 = cms.Path(process.generator*process.prodEE*process.prodHEF)
