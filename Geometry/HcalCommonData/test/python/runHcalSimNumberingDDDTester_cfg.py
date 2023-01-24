###############################################################################
# Way to use this:
#   cmsRun runHcalSimNumberingDDDTester_cfg.py geometry=Run3
#
#   Options for geometry Run3, Phase2
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "Run3",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: Run3, Phase2")

### get and parse the command line arguments
options.parseArguments()
print(options)

if (options.geometry == "Phase2"):
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process("HcalSimNumberingTest",Phase2C17I13M9)
    process.load('Geometry.HcalCommonData.testPhase2GeometryFine_cff')
else:
    from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD
    process = cms.Process("HcalSimNumberingTest",Run3_DDD)
    process.load('Configuration.Geometry.GeometryExtended2021_cff')

process.load('FWCore.MessageService.MessageLogger_cfi')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.HCalGeom=dict()

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.hpa = cms.EDAnalyzer("HcalSimNumberingTester")

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.hpa)
