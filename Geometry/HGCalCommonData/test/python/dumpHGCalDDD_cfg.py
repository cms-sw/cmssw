###############################################################################
# Way to use this:
#   cmsRun dumpHGCalDDD_cfg.py geometry=D88
#
#   Options for geometry D77, D83, D88, D92, D93
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
                  "geometry of operations: D77, D83, D88, D92, D93")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

if (options.geometry == "D83"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('DUMP',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D83Reco_cff')
    fileName = 'hgcalV15DDD.root'
elif (options.geometry == "D77"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('DUMP',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D77Reco_cff')
    fileName = 'hgcalV14DDD.root'
elif (options.geometry == "D92"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('DUMP',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D92Reco_cff')
    fileName = 'hgcalV17DDD.root'
elif (options.geometry == "D93"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('DUMP',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D93Reco_cff')
    fileName = 'hgcalV17NDDD.root'
else:
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('DUMP',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
    fileName = 'hgcalV16DDD.root'

print("Output file Name: ", fileName)

process.load('FWCore.MessageService.MessageLogger_cfi')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.G4cout=dict()
    process.MessageLogger.HGCalGeom=dict()

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.add_(cms.ESProducer("TGeoMgrFromDdd",
        verbose = cms.untracked.bool(False),
        level   = cms.untracked.int32(14)
))

process.dump = cms.EDAnalyzer("DumpSimGeometry",
                              outputFileName = cms.untracked.string(fileName))

process.p = cms.Path(process.dump)
