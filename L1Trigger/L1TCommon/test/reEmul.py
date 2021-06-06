from __future__ import print_function
#
#  reEmul.py  configurable test of L1T re-emulation
#
#  Example Usage:
#
#     cmsRun reEmul.py                 : stage-2 re-emulation on 2015 zero-bias data
#
#     cmsRun reEmul.py era=stage1      : stage-1 re-emulation on 2015 zero-bias data
#
#     cmsRun reEmul.py era=stage2 output=mu_stage2.root input=/store/data/Run2015D/DoubleMuon/RAW/v1/000/260/627/00000/004EF961-6082-E511-BFB0-02163E011BC4.root max=10
#                                      : stage-2 re-emulation on 2015 double muon data
#
#  Limitations:
#
#     - stage-1 re-emulation will complain about DT digi unpacking... harmless.  Will go away when we use GT for data + overrides for stage-1 emulation.
#
#     - stage-1 re-emulation does not put muons into the ntuple... will be fixed when legacy->upgrade converter is provided for muons.
#

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
from Configuration.Eras.Era_Run2_25ns_cff import Run2_25ns
from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
from Configuration.Eras.Modifier_stage1L1Trigger_cff import stage1L1Trigger
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
import os
import sys
import commands

options = VarParsing.VarParsing ('analysis')

options.register ('era',    'stage2',  VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string,  "The data taking Era: stage1 or stage2")
options.register ('output', 'DEFAULT', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string,  "The output file name")
options.register ('input',  '',        VarParsing.VarParsing.multiplicity.list,      VarParsing.VarParsing.varType.string,  "The input files")
options.register ('max',    '',        VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,     "The maximum number of events to process")
options.register ('skip',   '',        VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,     "The number of events to skip initially")

## options.input = '/store/data/Run2015D/ZeroBias/RAW/v1/000/260/627/00000/00A76FFA-0C82-E511-B441-02163E01450F.root'
options.input = '/store/data/Run2015D/ZeroBias1/RAW/v1/000/256/843/00000/FE8AD1BB-D05E-E511-B3A7-02163E01276B.root'
options.max  = 100 
options.skip = 0 

options.parseArguments()

if (options.era == 'stage1'):
    print("INFO: runnings L1T Stage-1 (2015) Re-Emulation")
    process = cms.Process("L1TReEmulation", Run2_25ns)
elif (options.era == 'stage2'):
    print("INFO: runnings L1T Stage-2 (2016) Re-Emulation")    
    process = cms.Process("L1TReEmulation", Run2_2016)
else:
    print("ERROR: unknown era:  ", options.era, "\n")
    exit(0)

if (options.output == 'DEFAULT'):
    tmp = cms.PSet(output = cms.string(""))
    stage1L1Trigger.toModify(tmp, output ='l1t_stage1.root')
    stage2L1Trigger.toModify(tmp, output ='l1t_stage2.root')
    options.output = tmp.output.value()
print("INFO: output:  ", options.output)

print("INFO: input:  ", options.input)
print("INFO: max:  ", options.max)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(50)
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(False))


process.source = cms.Source(
    'PoolSource',
    fileNames = cms.untracked.vstring(options.input)
)
if options.skip > 0:
    process.source.skipEvents = cms.untracked.uint32(options.skip)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.max))

# PostLS1 geometry used
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2015_cff')
############################
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag

#   For stage-1, we are re-emulating L1T based on the conditions in the GT, so
#   best for now to use MC GT, even when running over a data file, and just
#   ignore the main DT TP emulator warnings...  (In future we'll override only
#   L1T emulator related conditions, so you can use a data GT)
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
# Note:  For stage-2, all conditions are overriden by local config file.  Use data tag: 

stage2L1Trigger.toReplaceWith(process.GlobalTag,
    #GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
    GlobalTag(process.GlobalTag, 'auto:run2_data', '').clone(
    toGet = cms.VPSet(
        cms.PSet(record = cms.string("DTCCBConfigRcd"),
                 tag = cms.string("DTCCBConfig_NOSingleL_V05_mc"),
                 connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
                 ),
#        cms.PSet(record = cms.string("DTKeyedConfigListRcd"),
#                 tag = cms.string("DTKeyedConfigContainer_NOSingleL_V05_mc"),
#                 connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
#                 ),
        cms.PSet(record = cms.string("DTT0Rcd"),
                 tag = cms.string("DTt0_STARTUP_V02_mc"),
                 connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
                 ),
        cms.PSet(record = cms.string("DTTPGParametersRcd"),
                 tag = cms.string("DTTPGParameters_V01_mc"),
                 connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
                 ),
        )
))

#### Sim L1 Emulator Sequence:
process.load('Configuration.StandardSequences.RawToDigi_cff')
#process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('L1Trigger.Configuration.L1TReEmulateFromRAW_cff')
process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")





import L1Trigger.L1TCommon.l1tSummaryStage1Digis_cfi
import L1Trigger.L1TCommon.l1tSummaryStage2Digis_cf
process.l1tSummaryA = L1Trigger.L1TCommon.l1tSummaryStage1Digis_cfi.l1tSummaryStage1Digis.clone()
stage2L1Trigger.toReplaceWith(process.l1tSummaryA, L1Trigger.L1TCommon.l1tSummaryStage2Digis_cfi.l1tSummaryStage2Digis.clone())

import L1Trigger.L1TCommon.l1tSummaryStage1SimDigis_cfi
import L1Trigger.L1TCommon.l1tSummaryStage2SimDigis_cfi
process.l1tSummaryB = L1Trigger.L1TCommon.l1tSummaryStage1SimDigis_cfi.l1tSummaryStage1SimDigis.clone()
stage2L1Trigger.toReplaceWith(process.l1tSummaryB, L1Trigger.L1TCommon.l1tSummaryStage2SimDigis_cfi.l1tSummaryStage2SimDigis.clone())

# Additional output definition
# TTree output file
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string(options.output)


# enable debug message logging for our modules
process.MessageLogger.L1TCaloEvents=dict()
process.MessageLogger.L1TGlobalEvents=dict()
process.MessageLogger.l1t=dict()
process.MessageLogger.Global=dict()
process.MessageLogger.suppressInfo = cms.untracked.vstring('Geometry', 'AfterSource')

# gt analyzer
process.l1tGlobalAnalyzer = cms.EDAnalyzer('L1TGlobalAnalyzer',
    doText = cms.untracked.bool(True),
    dmxEGToken = cms.InputTag("None"),
    dmxTauToken = cms.InputTag("None"),
    dmxJetToken = cms.InputTag("None"),
    dmxEtSumToken = cms.InputTag("None"),
    muToken = cms.InputTag("simGmtStage2Digis"),
    egToken = cms.InputTag("simCaloStage2Digis"),
    tauToken = cms.InputTag("simCaloStage2Digis"),
    jetToken = cms.InputTag("simCaloStage2Digis"),
    etSumToken = cms.InputTag("simCaloStage2Digis"),
    gtAlgToken = cms.InputTag("None"),
    emulDxAlgToken = cms.InputTag("simGlobalStage2Digis"),
    emulGtAlgToken = cms.InputTag("simGlobalStage2Digis")
)

process.l1EventTree = cms.EDAnalyzer("L1EventTreeProducer",
                                     hltSource            = cms.InputTag("TriggerResults::HLT"),
                                     puMCFile             = cms.untracked.string(""),
                                     puDataFile           = cms.untracked.string(""),
                                     puMCHist             = cms.untracked.string(""),
                                     puDataHist           = cms.untracked.string(""),
                                     
                                     useAvgVtx            = cms.untracked.bool(True),
                                     maxAllowedWeight     = cms.untracked.double(-1)                                     
)

# Stage-1 Ntuple will not contain muons, and might (investigating) miss some Taus.  (Work underway)
process.l1UpgradeTree = cms.EDAnalyzer("L1UpgradeTreeProducer",
    egToken      = cms.untracked.InputTag("simCaloStage1FinalDigis"),
    tauTokens    = cms.untracked.VInputTag("simCaloStage1FinalDigis:rlxTaus","simCaloStage1FinalDigis:isoTaus"),
    jetToken     = cms.untracked.InputTag("simCaloStage1FinalDigis"),
    muonToken    = cms.untracked.InputTag("None"),
    sumToken     = cms.untracked.InputTag("simCaloStage1FinalDigis",""),
    maxL1Upgrade = cms.uint32(60)
)
stage2L1Trigger.toModify(process.l1UpgradeTree,
    egToken      = "simCaloStage2Digis",
    tauTokens    = ["simCaloStage2Digis"],
    jetToken     = "simCaloStage2Digis",
    muonToken    = "simGmtStage2Digis",
    sumToken     = "simCaloStage2Digis"
)


process.L1TSeq = cms.Sequence(   process.RawToDigi        
                                   + process.L1TReEmulateFromRAW
#                                   + process.dumpED
#                                   + process.dumpES
#                                   + process.l1tSummaryA
# Comment this next module to silence per event dump of L1T objects:
                                   + process.l1tSummaryB
#                                   + process.l1tGlobalAnalyzer
                                   + process.l1UpgradeTree
                                   + process.l1EventTree                                 
)

process.L1TPath = cms.Path(process.L1TSeq)

process.schedule = cms.Schedule(process.L1TPath)

# Re-emulating, so don't unpack L1T output, might not even exist...
# Also, remove uneeded unpackers for speed.
stage2L1Trigger.toReplaceWith(process.L1TSeq, process.L1TSeq.copyAndExclude([
    process.gmtStage2Digis,
    process.caloStage2Digis,
    process.gtStage2Digis,
    process.siPixelDigis,
    process.siStripDigis,
    process.castorDigis,
    process.scalersRawToDigi,
    process.tcdsDigis
]))
stage1L1Trigger.toReplaceWith(process.L1TSeq, process.L1TSeq.copyAndExclude([
    process.caloStage1Digis,
    process.caloStage1FinalDigis,
    process.gtDigis,
    process.siPixelDigis,
    process.siStripDigis,
    process.castorDigis,
    process.scalersRawToDigi,
    process.tcdsDigis
]))

print(process.L1TSeq)
print(process.L1TReEmulateFromRAW)
#processDumpFile = open('config.dump', 'w')
#print >> processDumpFile, process.dumpPython()


#process.out = cms.OutputModule("PoolOutputModule", 
#   fileName = cms.untracked.string("l1tbmtf.root")
#)
#process.output_step = cms.EndPath(process.out)
#process.schedule.extend([process.output_step])
