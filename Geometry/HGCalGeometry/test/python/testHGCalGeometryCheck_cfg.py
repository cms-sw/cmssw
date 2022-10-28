###############################################################################
# Way to use this:
#   cmsRun testHGCalGeometryCheck_cfg.py geometry=D77
#
#   Options for geometry D49, D68, D77, D83, D84, D88, D92
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D88",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D49, D68, D84, D77, D83, D88, D92")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

if (options.geometry == "D49"):
    from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
    process = cms.Process('HGCGeomCheck',Phase2C9)
    process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
    fileName = 'HGCGeomStudyV11.root'
elif (options.geometry == "D68"):
    from Configuration.Eras.Era_Phase2C12_cff import Phase2C12
    process = cms.Process('HGCGeomCheck',Phase2C12)
    process.load('Configuration.Geometry.GeometryExtended2026D68Reco_cff')
    fileName = 'HGCGeomStudyV12.root'
elif (options.geometry == "D83"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('HGCGeomCheck',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D83Reco_cff')
    fileName = 'HGCGeomStudyV15.root'
elif (options.geometry == "D84"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('HGCGeomCheck',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D84Reco_cff')
    fileName = 'HGCGeomStudyV13.root'
elif (options.geometry == "D88"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('HGCGeomCheck',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
    fileName = 'HGCGeomStudyV16.root'
elif (options.geometry == "D92"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('HGCGeomCheck',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D92Reco_cff')
    fileName = 'HGCGeomStudyV17.root'
else:
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('HGCGeomCheck',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D77Reco_cff')
    fileName = 'HGCGeomStudyV14.root'

print("Output file: ", fileName)

process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load('Geometry.HGCalGeometry.hgcalGeometryCheck_cfi')
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

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(fileName),
                                   closeFileFast = cms.untracked.bool(True)
                                   )

#process.hgcalGeometryCheck.verbosity = True

process.p1 = cms.Path(process.generator*process.hgcalGeometryCheck)
