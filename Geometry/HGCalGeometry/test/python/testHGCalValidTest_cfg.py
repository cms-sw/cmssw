import FWCore.ParameterSet.Config as cms

geomName = "Run4D110"
geomFile = "Configuration.Geometry.GeometryExtended" + geomName + "Reco_cff"
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
GLOBAL_TAG, ERA = _settings.get_era_and_conditions(geomName)

print("Geometry Name:   ", geomName)
print("Geom file Name:  ", geomFile)
print("Global Tag Name: ", GLOBAL_TAG)
print("Era Name:        ", ERA)

process = cms.Process('PROD',ERA)

process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load(geomFile)
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

process.load("Geometry.HGCalGeometry.hgcalValidTestHEB_cfi")

process.hgcalValidTestHEF = process.hgcalValidTestHEB.clone(
    detector   = "HGCalHESiliconSensitive",
)

process.hgcalValidTestEE = process.hgcalValidTestHEB.clone(
    detector   = "HGCalEESensitive"
)

#process.p1 = cms.Path(process.generator*process.hgcalValidTestEE*process.hgcalValidTestHEF*process.hgcalValidTestHEB)
process.p1 = cms.Path(process.generator*process.hgcalValidTestHEB)
