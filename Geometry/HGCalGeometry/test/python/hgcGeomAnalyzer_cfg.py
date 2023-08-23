import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.StandardSequences.Eras import eras
from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

options = VarParsing()
options.register ("geom", "",  VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.parseArguments()

fileName = "geom_output_"+options.geom

process = cms.Process("demo",eras.Phase2C17I13M9)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
if options.geom == 'v16':
    process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
elif options.geom == 'v17':
    process.load('Configuration.Geometry.GeometryExtended2026D92Reco_cff')
else:
    raise Exception('UNKNOWN GEOMETRY!')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

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
