###############################################################################
# Way to use this:
#   cmsRun testHGCalCellDumpDDD_cfg.py geometry=D86
#
#   Options for geometry D86, D88, D92, D93
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D86",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D86, D88, D92, D93")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

if (options.geometry == "D88"):
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process('PROD',Phase2C17I13M9)
    process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
elif (options.geometry == "D92"):
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process('PROD',Phase2C17I13M9)
    process.load('Configuration.Geometry.GeometryExtended2026D92Reco_cff')
elif (options.geometry == "D93"):
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process('PROD',Phase2C17I13M9)
    process.load('Configuration.Geometry.GeometryExtended2026D93Reco_cff')
else:
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process('PROD',Phase2C17I13M9)
    process.load('Configuration.Geometry.GeometryExtended2026D86Reco_cff')

process.load("Geometry.HGCalGeometry.hgcalGeometryDump_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalGeom=dict()

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )
process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.hgcalGeometryDump)
