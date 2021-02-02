import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

#process.load("Geometry.HcalCommonData.testGeometry17bXML_cfi")
#process.load("Geometry.HcalCommonData.hcalDDConstants_cff")
#process.load("Geometry.HcalEventSetup.hcalTopologyIdeal_cfi")
process.load('Configuration.Geometry.GeometryExtended2023D41Reco_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.HcalGeom=dict()
    process.MessageLogger.CaloTopology=dict()

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

process.load("Geometry.CaloTopology.hcalDetId2DenseTester_cfi")
process.hcalDetId2DenseTester.fileName = "hcaldetid2dense.txt"
process.hcalDetId2DenseTester.testCalib = True

process.p1 = cms.Path(process.generator*process.hcalDetId2DenseTester)
