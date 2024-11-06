###############################################################################
# Way to use this:
#   cmsRun hgcGeomAnalyzer_cfg.py geom=v17
#
#   Options for geometry v16, v17, v18
#
###############################################################################
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.StandardSequences.Eras import eras
from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

options = VarParsing()
options.register ("geom",
                  "v17",
                  VarParsing.multiplicity.singleton, VarParsing.varType.string,
                  "geom of operations: v16, v17, v18")

options.parseArguments()

process = cms.Process("demo",eras.Phase2C17I13M9)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
if options.geom == 'v16':
    geomFile = 'Configuration.Geometry.GeometryExtendedRun4D100Reco_cff'
elif options.geom == 'v17':
    geomFile = 'Configuration.Geometry.GeometryExtendedRun4D110Reco_cff'
elif options.geom == 'v18':
    geomFile = 'Configuration.Geometry.GeometryExtendedRun4D104Reco_cff'
else:
    geomFile = 'UNKNOWN GEOMETRY!'
    raise Exception(geomFile)

fileName = "geom_output_"+options.geom

print("Geometry file: ", geomFile)
print("Output   file: ", fileName)

process.load(geomFile)
 
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T25', '')

process.load('FWCore.MessageService.MessageLogger_cfi')
if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalGeom=dict()
    process.MessageLogger.HGCalGeomX=dict()

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
process.source = cms.Source("EmptySource")

process.plotter = cms.EDAnalyzer("HGCGeomAnalyzer",
    fileName = cms.string(fileName+".txt")
    )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(fileName+".root")
)

process.p = cms.Path(process.plotter)
