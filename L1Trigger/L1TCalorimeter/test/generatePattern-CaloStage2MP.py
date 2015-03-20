# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: SingleElectronPt10_cfi.py -s GEN,SIM,DIGI,L1 --pileup=NoPileUp --geometry DB --conditions=auto:startup -n 1 --no_exec
import FWCore.ParameterSet.Config as cms


# options
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing('analysis')
options.register('skipEvents',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Number of events to skip")
#options.register('mpFramesPerEvent',
#                 40,
#                 VarParsing.VarParsing.multiplicity.singleton,
#                 VarParsing.VarParsing.varType.int,
#                 "MP frames per event")
#options.register('mpOffset',
#                 0,
#                 VarParsing.VarParsing.multiplicity.singleton,
#                 VarParsing.VarParsing.varType.int,
#                 "MP offset (frames)")
#options.register('nMP',
#                 11,
#                 VarParsing.VarParsing.multiplicity.singleton,
#                 VarParsing.VarParsing.varType.int,
#                 "Number of MPs")
options.register('dump',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Enable print out")
options.register('debug',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Enable debug data")
options.register('data',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Run on data not simulation")
                 
options.parseArguments()

if (options.maxEvents == -1):
    options.maxEvents = 1



process = cms.Process('L1TPATTERN')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.Geometry.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

# Input source
process.source = cms.Source ("PoolSource",
    fileNames=cms.untracked.vstring(options.inputFiles)
)

process.options = cms.untracked.PSet(

)

# global tag
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')


# enable debug message logging for our modules

process.MessageLogger.categories.append('L1TCaloEvents')

process.MessageLogger.suppressInfo = cms.untracked.vstring('Geometry', 'AfterSource')

#if (options.dump):
#    process.MessageLogger.infos.placeholder = cms.untracked.bool(False)
#    process.MessageLogger.infos.INFO = cms.untracked.PSet(limit = cms.untracked.int32(0))
#    process.MessageLogger.infos.L1TCaloEvents = cms.untracked.PSet(
#      optionalPSet = cms.untracked.bool(True),
#      limit = cms.untracked.int32(10000)
#    )

if (options.debug):
#    process.MessageLogger.debugModules = cms.untracked.vstring('L1TRawToDigi:caloStage2Digis', 'MP7BufferDumpToRaw:stage2MPRaw', 'MP7BufferDumpToRaw:stage2DemuxRaw')
    process.MessageLogger.debugModules = cms.untracked.vstring('*')
    process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')
process.MessageLogger = cms.Service(
    "MessageLogger",
    destinations   = cms.untracked.vstring(
	'detailedInfo',
	'critical'
    ),
    detailedInfo   = cms.untracked.PSet(
	threshold  = cms.untracked.string('DEBUG') 
    ),
    debugModules = cms.untracked.vstring(
	'caloStage2Layer1Digis',
	'caloStage2Digis'
    )
)


# upgrade calo stage 2
#process.load('L1Trigger.L1TCalorimeter.caloStage2Params_cfi')
#process.load('L1Trigger.L1TCalorimeter.L1TCaloStage2_cff')
#process.caloStage2Layer1Digis.ecalToken = cms.InputTag("simEcalTriggerPrimitiveDigis")
#process.caloStage2Layer1Digis.hcalToken = cms.InputTag("simHcalTriggerPrimitiveDigis")

process.load('L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi')
if (options.data):
    process.l1tStage2CaloAnalyzer.towerToken = cms.InputTag("caloStage2Digis")
process.l1tStage2CaloAnalyzer.clusterToken = cms.InputTag("None")
process.l1tStage2CaloAnalyzer.mpEGToken = cms.InputTag("None")
process.l1tStage2CaloAnalyzer.mpTauToken = cms.InputTag("None")


process.load('L1Trigger.L1TCalorimeter.l1tStage2InputPatternWriter_cfi')
process.l1tStage2InputPatternWriter.filename = cms.untracked.string("pattern.txt")
if (options.data):
    process.l1tStage2InputPatternWriter.towerToken = cms.InputTag("caloStage2Digis")

process.path = cms.Path(
  process.l1tStage2InputPatternWriter
)
