import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing('analysis')
options.register('globaltag',	'', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, '') 
options.register('run',	'', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, '') 
options.register('lutXML1',	'', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, '') 
options.register('lutXML2',	'', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, '') 
options.register('verbosity',	'', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, '') 

options.parseArguments()

process = cms.Process("LutDiff")

process.load("Configuration.Geometry.GeometryDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = options.globaltag 

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
process.source = cms.Source("EmptySource")
process.source.firstRun = cms.untracked.uint32(options.run)

process.diff  = cms.EDAnalyzer("HcalLutComparer",
    lutXML1  = cms.string(options.lutXML1),
    lutXML2  = cms.string(options.lutXML2),
    verbosity = cms.uint32(options.verbosity),
)
process.p = cms.Path(process.diff)
