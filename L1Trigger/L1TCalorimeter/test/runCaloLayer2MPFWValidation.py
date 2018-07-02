import FWCore.ParameterSet.Config as cms

# options
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing('analysis')
options.register('skipEvents',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Number of events to skip")
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
options.register('dumpTowers',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 'Dump all towers in text form when a problem is found')
options.register('dumpWholeEvent',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 'Dump all event contents in text when a problem is found')

options.parseArguments()


process = cms.Process('CaloLayer2MPFWValidation')

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
    outputCommands = cms.untracked.vstring(
        "drop *",
        "keep *_*_dataJet_*",
        "keep *_*_emulJet_*",
        "keep *_*_dataEg_*",
        "keep *_*_emulEg_*",
        "keep *_*_dataTau_*",
        "keep *_*_emulTau_*",
        "keep *_*_dataEtSum_*",
        "keep *_*_emulEtSum_*",
        "keep *_*_dataCaloTower_*",
        "keep *_*_emulCaloTower_*",
        ),
    fileName = cms.untracked.string('l1tCaloLayer2CompEDM.root')
)

# Additional output definition
# TTree output file
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string('l1tCalo_2016_simHistos.root')


# enable debug message logging for our modules
process.MessageLogger = cms.Service(
    "MessageLogger",
    threshold  = cms.untracked.string('ERROR'),
    categories = cms.untracked.vstring('L1T'),
    destinations = cms.untracked.vstring('calol2_mp_fw_emul_differences'),
    debugModules = cms.untracked.vstring('L1Trigger.L1TCalorimeter.l1tStage2CaloLayer2Comp_cfi')
)


# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')

# emulator
process.load('L1Trigger.L1TCalorimeter.simCaloStage2Digis_cfi')
process.simCaloStage2Digis.useStaticConfig = True
process.simCaloStage2Digis.towerToken = cms.InputTag("caloStage2Digis","CaloTower")

# emulator ES
process.load('L1Trigger.L1TCalorimeter.caloParams_2018_v1_3_cfi')

# histograms
process.load('L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi')
process.l1tStage2CaloAnalyzer.doEvtDisp = options.evtDisp
process.l1tStage2CaloAnalyzer.mpBx = options.selMPBx
process.l1tStage2CaloAnalyzer.dmxBx = options.selDemuxBx
process.l1tStage2CaloAnalyzer.allBx = options.selAllBx
process.l1tStage2CaloAnalyzer.clusterToken = cms.InputTag("None")
process.l1tStage2CaloAnalyzer.towerToken = cms.InputTag("l1tStage2CaloLayer2Comp","emulCaloTower")
process.l1tStage2CaloAnalyzer.mpEGToken = cms.InputTag("l1tStage2CaloLayer2Comp", "emulEg")
process.l1tStage2CaloAnalyzer.mpTauToken = cms.InputTag("l1tStage2CaloLayer2Comp", "emulTau")
process.l1tStage2CaloAnalyzer.mpJetToken = cms.InputTag("l1tStage2CaloLayer2Comp", "emulJet")
process.l1tStage2CaloAnalyzer.mpEtSumToken = cms.InputTag("l1tStage2CaloLayer2Comp", "emulEtSum")

import L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi
process.l1tCaloStage2HwHistos =  L1Trigger.L1TCalorimeter.l1tStage2CaloAnalyzer_cfi.l1tStage2CaloAnalyzer.clone()
process.l1tStage2CaloAnalyzer.doEvtDisp = options.evtDisp
process.l1tStage2CaloAnalyzer.mpBx = options.selMPBx
process.l1tStage2CaloAnalyzer.dmxBx = options.selDemuxBx
process.l1tStage2CaloAnalyzer.allBx = options.selAllBx
process.l1tCaloStage2HwHistos.clusterToken = cms.InputTag("None")
process.l1tCaloStage2HwHistos.towerToken = cms.InputTag("l1tStage2CaloLayer2Comp", "dataCaloTower")
process.l1tCaloStage2HwHistos.mpEGToken = cms.InputTag("l1tStage2CaloLayer2Comp", "dataEg")
process.l1tCaloStage2HwHistos.mpTauToken = cms.InputTag("l1tStage2CaloLayer2Comp", "dataTau")
process.l1tCaloStage2HwHistos.mpJetToken = cms.InputTag("l1tStage2CaloLayer2Comp", "dataJet")
process.l1tCaloStage2HwHistos.mpEtSumToken = cms.InputTag("l1tStage2CaloLayer2Comp", "dataEtSum")

# Event by event comparisons
process.load('L1Trigger.L1TCalorimeter.l1tStage2CaloLayer2Comp_cfi')
process.l1tStage2CaloLayer2Comp.dumpTowers = options.dumpTowers
process.l1tStage2CaloLayer2Comp.dumpWholeEvent = options.dumpWholeEvent

# Path and EndPath definitions
process.path = cms.Path(
    process.simCaloStage2Digis
    + process.l1tStage2CaloLayer2Comp
    + process.l1tStage2CaloAnalyzer
    + process.l1tCaloStage2HwHistos
)

#if (not options.doMP):
#    process.path.remove(process.stage2MPRaw)

#if (not options.-doDemux):
#    process.path.remove(process.stage2DemuxRaw)

process.out = cms.EndPath(
    process.output
)

