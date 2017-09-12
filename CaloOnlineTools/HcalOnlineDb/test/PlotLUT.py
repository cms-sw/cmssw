import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing('analysis')
options.register('globaltag',	'', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, '') 
options.register('run',	        '', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, '') 
options.register('inputDir',	'', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, '') 
options.register('plotsDir',	'', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, '') 
options.register('tags',	'', VarParsing.VarParsing.multiplicity.list, VarParsing.VarParsing.varType.string, '') 
options.register('gains',	'', VarParsing.VarParsing.multiplicity.list, VarParsing.VarParsing.varType.string, '') 
options.register('respcorrs',	'', VarParsing.VarParsing.multiplicity.list, VarParsing.VarParsing.varType.string, '') 
options.register('pedestals',	'', VarParsing.VarParsing.multiplicity.list, VarParsing.VarParsing.varType.string, '') 
options.register('quality',	'', VarParsing.VarParsing.multiplicity.list, VarParsing.VarParsing.varType.string, '') 
options.parseArguments()

process = cms.Process("LutPlot")

process.load("Configuration.Geometry.GeometryDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = options.globaltag 

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
process.source = cms.Source("EmptySource")
process.source.firstRun = cms.untracked.uint32(options.run)

process.plot  = cms.EDAnalyzer("HcalLutAnalyzer",
    inputDir  = cms.string(options.inputDir),
    plotsDir  = cms.string(options.plotsDir),
    tags      = cms.vstring(options.tags),
    gains     = cms.vstring(options.gains),
    respcorrs = cms.vstring(options.respcorrs),
    pedestals = cms.vstring(options.pedestals),
    quality   = cms.vstring(options.quality),
    Zmin      = cms.double(0),
    Zmax      = cms.double(10),
    Ymin      = cms.double(0.7),
    Ymax      = cms.double(1.3),
    Pmin      = cms.double(0.9),
    Pmax      = cms.double(1.1),
)
process.p = cms.Path(process.plot)

