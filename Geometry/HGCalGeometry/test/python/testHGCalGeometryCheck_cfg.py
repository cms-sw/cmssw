import FWCore.ParameterSet.Config as cms

#from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
#process = cms.Process('PROD',Phase2C9)
#process.load('Configuration.Geometry.GeometryExtended2026D49_cff')
#process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')

from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
process = cms.Process('PROD',Phase2C11)
process.load('Configuration.Geometry.GeometryExtended2026D71_cff')
process.load('Configuration.Geometry.GeometryExtended2026D71Reco_cff')

process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load('Geometry.HGCalGeometry.hgcalGeometryCheck_cfi')
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

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('hgcGeomStudy.root'),
                                   closeFileFast = cms.untracked.bool(True)
                                   )

#process.hgcalGeometryCheck.verbosity = True

process.p1 = cms.Path(process.generator*process.hgcalGeometryCheck)
