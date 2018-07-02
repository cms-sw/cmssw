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
options.register('dump',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Print RAW data")
options.register('doLayer1',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Run layer 1 module")
options.register('doLayer2',
                 True,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Run layer 2 module")
options.register('selMPBx',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Select MP readout Bx")
options.register('selDemuxBx',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Select Demux readout Bx")
options.register('selAllBx',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Run over all Bx in readout for MP and demux")
options.register('evtDisp',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 'Produce histos for individual events')

                 
options.parseArguments()


process = cms.Process('L1Emulator')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.Geometry.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

# Input source
#inFile = 'file:l1tCalo_2016_EDM.root'
process.source = cms.Source("PoolSource",
    fileNames=cms.untracked.vstring(options.inputFiles),
    skipEvents=cms.untracked.uint32(options.skipEvents)
)

process.options = cms.untracked.PSet(
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)


# Output definition
process.output = cms.OutputModule(
    "PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *"),
    fileName = cms.untracked.string('l1tCalo_2016_simEDM.root')
)

# Additional output definition
# TTree output file
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string('l1tCalo_2016_simHistos.root')


# enable debug message logging for our modules
process.MessageLogger = cms.Service(
    "MessageLogger",
    threshold  = cms.untracked.string('DEBUG'),
    categories = cms.untracked.vstring('L1T'),
    debugModules = cms.untracked.vstring('*')
#        'mp7BufferDumpToRaw',
#        'l1tDigis',
#        'caloStage1Digis'
#    )
)


# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')


# emulator
process.load('L1Trigger.L1TCalorimeter.simCaloStage2Layer1Digis_cfi')
process.load('L1Trigger.L1TCalorimeter.simCaloStage2Digis_cfi')
process.simCaloStage2Digis.useStaticConfig = True
process.simCaloStage2Layer1Digis.ecalToken = cms.InputTag("ecalDigis:EcalTriggerPrimitives")
process.simCaloStage2Layer1Digis.hcalToken = cms.InputTag("hcalDigis")

# emulator ES
process.load('L1Trigger.L1TCalorimeter.caloParams_2018_v1_3_cfi')

# histograms
process.load('L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi')
process.l1tStage2CaloAnalyzer.doEvtDisp = options.evtDisp
process.l1tStage2CaloAnalyzer.mpBx = options.selMPBx
process.l1tStage2CaloAnalyzer.dmxBx = options.selDemuxBx
process.l1tStage2CaloAnalyzer.allBx = options.selAllBx
process.l1tStage2CaloAnalyzer.towerToken = cms.InputTag("simCaloStage2Digis", "MP")
process.l1tStage2CaloAnalyzer.clusterToken = cms.InputTag("None")
process.l1tStage2CaloAnalyzer.mpEGToken = cms.InputTag("simCaloStage2Digis", "MP")
process.l1tStage2CaloAnalyzer.mpTauToken = cms.InputTag("simCaloStage2Digis", "MP")
process.l1tStage2CaloAnalyzer.mpJetToken = cms.InputTag("simCaloStage2Digis", "MP")
process.l1tStage2CaloAnalyzer.mpEtSumToken = cms.InputTag("simCaloStage2Digis", "MP")
process.l1tStage2CaloAnalyzer.egToken = cms.InputTag("simCaloStage2Digis")
process.l1tStage2CaloAnalyzer.tauToken = cms.InputTag("simCaloStage2Digis")
process.l1tStage2CaloAnalyzer.jetToken = cms.InputTag("simCaloStage2Digis")
process.l1tStage2CaloAnalyzer.etSumToken = cms.InputTag("simCaloStage2Digis")

import L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi
process.l1tCaloStage2HwHistos =  L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi.l1tStage2CaloAnalyzer.clone()
process.l1tStage2CaloAnalyzer.doEvtDisp = options.evtDisp
process.l1tStage2CaloAnalyzer.mpBx = options.selMPBx
process.l1tStage2CaloAnalyzer.dmxBx = options.selDemuxBx
process.l1tStage2CaloAnalyzer.allBx = options.selAllBx
process.l1tCaloStage2HwHistos.towerToken = cms.InputTag("caloStage2Digis","CaloTower")
process.l1tCaloStage2HwHistos.clusterToken = cms.InputTag("None")
process.l1tCaloStage2HwHistos.mpEGToken = cms.InputTag("caloStage2Digis", "MP")
process.l1tCaloStage2HwHistos.mpTauToken = cms.InputTag("caloStage2Digis","MP")
process.l1tCaloStage2HwHistos.mpJetToken = cms.InputTag("caloStage2Digis", "MP")
process.l1tCaloStage2HwHistos.mpEtSumToken = cms.InputTag("caloStage2Digis", "MP")
process.l1tCaloStage2HwHistos.egToken = cms.InputTag("caloStage2Digis","EGamma")
process.l1tCaloStage2HwHistos.tauToken = cms.InputTag("caloStage2Digis","Tau")
process.l1tCaloStage2HwHistos.jetToken = cms.InputTag("caloStage2Digis","Jet")
process.l1tCaloStage2HwHistos.etSumToken = cms.InputTag("caloStage2Digis","EtSum")

# Path and EndPath definitions
process.path = cms.Path(
    process.ecalDigis
    +process.hcalDigis
    +process.simCaloStage2Layer1Digis
    +process.simCaloStage2Digis
    +process.l1tStage2CaloAnalyzer
    +process.l1tCaloStage2HwHistos
)

if (not options.doLayer1):
    process.path.remove(process.ecalDigis)
    process.path.remove(process.hcalDigis)
    process.path.remove(process.simCaloStage2Layer1Digis)
    process.simCaloStage2Digis.towerToken = cms.InputTag("caloStage2Digis","CaloTower")

if (not options.doLayer2):
    process.path.remove(process.simCaloStage2Digis)


#if (not options.doMP):
#    process.path.remove(process.stage2MPRaw)

#if (not options.-doDemux):
#    process.path.remove(process.stage2DemuxRaw)

process.out = cms.EndPath(
    process.output
)

