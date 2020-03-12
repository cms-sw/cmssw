import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

#process.load("Geometry.HGCalCommonData.testHGCV8XML_cfi")
#process.load("Geometry.HGCalCommonData.hgcalV6NumberingInitialization_cfi")
#process.load("Geometry.HGCalCommonData.hgcalV6ParametersInitialization_cfi")
#process.load("Geometry.CaloEventSetup.HGCalV6Topology_cfi")
process.load("Geometry.HGCalCommonData.testHGCXML_cfi")
process.load("Geometry.HGCalCommonData.hgcalParametersInitialization_cfi")
process.load("Geometry.HGCalCommonData.hgcalNumberingInitialization_cfi")
process.load("Geometry.CaloEventSetup.HGCalV9Topology_cfi")
process.load("Geometry.HGCalGeometry.HGCalGeometryESProducer_cfi")
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

process.prodEE = cms.EDAnalyzer("HGCalSizeTester",
                                detector   = cms.string("HGCalEESensitive"),
                                )

process.prodHEF = process.prodEE.clone(
    detector   = "HGCalHESiliconSensitive",
)

process.prodHEB = process.prodEE.clone(
    detector   = "HGCalHEScintillatorSensitive",
)

#process.p1 = cms.Path(process.generator*process.prodEE*process.prodHEF)
#process.p1 = cms.Path(process.prodHEB)
process.p1 = cms.Path(process.generator*process.prodEE*process.prodHEF*process.prodHEB)
