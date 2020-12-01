from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os
import sys
import commands
##############################################################################
# customisations for L1T utilities
#
# customisations which add utilities features such as debugging of L1T,
#    summary module, etc.
#
##############################################################################

# Unpack Stage-2 GT and GMT
def L1TTurnOffGtAndGmtEmulation(process):
    cutlist=['simDtTriggerPrimitiveDigis','simCscTriggerPrimitiveDigis','simTwinMuxDigis','simBmtfDigis','simEmtfDigis','simOmtfDigis','simGmtCaloSumDigis','simMuonQualityAdjusterDigis','simGmtStage2Digis','simGtStage2Digis']
    for b in cutlist:
        process.SimL1Emulator.remove(getattr(process,b))
    return process

# Unpack Stage-2 GT and GMT
def L1TTurnOffUnpackStage2GtAndGmt(process):
    cutlist=['gtStage2Digis','gmtStage2Digis']
    for b in cutlist:
        process.L1TRawToDigi.remove(getattr(process,b))
    return process

# Unpack Stage-2 GT and GMT
def L1TTurnOffUnpackStage2GtGmtAndCalo(process):
    cutlist=['gtStage2Digis','gmtStage2Digis','caloStage2Digis']
    for b in cutlist:
        process.L1TRawToDigi.remove(getattr(process,b))
    return process

def L1TStage1DigisSummary(process):
    print("L1T INFO:  will dump a summary of unpacked Stage1 content to screen.")
    process.load('L1Trigger.L1TCommon.l1tSummaryStage1Digis_cfi')
    process.l1tstage1summary = cms.Path(process.l1tSummaryStage1Digis)
    process.schedule.append(process.l1tstage1summary)
    return process

def L1TStage2DigisSummary(process):
    print("L1T INFO:  will dump a summary of unpacked Stage2 content to screen.")
    process.load('L1Trigger.L1TCommon.l1tSummaryStage2Digis_cfi')
    process.l1tstage2summary = cms.Path(process.l1tSummaryStage2Digis)
    process.schedule.append(process.l1tstage2summary)
    return process

def L1TStage1SimDigisSummary(process):
    print("L1T INFO:  will dump a summary of simulated Stage1 content to screen.")
    process.load('L1Trigger.L1TCommon.l1tSummaryStage1SimDigis_cfi')
    process.l1tsimstage1summary = cms.Path(process.l1tSummaryStage1SimDigis)
    process.schedule.append(process.l1tsimstage1summary)
    return process

def L1TStage2SimDigisSummary(process):
    print("L1T INFO:  will dump a summary of simulated Stage2 content to screen.")
    process.load('L1Trigger.L1TCommon.l1tSummaryStage2SimDigis_cfi')
    process.l1tsimstage2summary = cms.Path(process.l1tSummaryStage2SimDigis)
    process.schedule.append(process.l1tsimstage2summary)
    return process

def L1TGlobalDigisSummary(process):
    print("L1T INFO:  will dump a summary of unpacked L1T Global output to screen.")
    process.l1tGlobalSummary = cms.EDAnalyzer(
        'L1TGlobalSummary',
        AlgInputTag = cms.InputTag("gtStage2Digis"),
        ExtInputTag = cms.InputTag("gtStage2Digis"),
        DumpTrigResults = cms.bool(False), # per event dump of trig results
        DumpTrigSummary = cms.bool(True), # pre run dump of trig results
        )
    process.l1tglobalsummary = cms.Path(process.l1tGlobalSummary)
    process.schedule.append(process.l1tglobalsummary)
    return process

def L1TGlobalMenuXML(process):
    process.load('L1Trigger.L1TGlobal.GlobalParameters_cff')
    process.load('L1Trigger.L1TGlobal.TriggerMenu_cff')
    process.TriggerMenu.L1TriggerMenuFile = cms.string('L1Menu_Collisions2016_v2c.xml')
    return process

def L1TGlobalSimDigisSummary(process):
    print("L1T INFO:  will dump a summary of simulated L1T Global output to screen.")
    process.l1tSimGlobalSummary = cms.EDAnalyzer(
        'L1TGlobalSummary',
        AlgInputTag = cms.InputTag("simGtStage2Digis"),
        ExtInputTag = cms.InputTag("simGtStage2Digis"),
        DumpTrigResults = cms.bool(False), # per event dump of trig results
        DumpTrigSummary = cms.bool(True), # pre run dump of trig results
        )
    process.l1tsimglobalsummary = cms.Path(process.l1tSimGlobalSummary)
    process.schedule.append(process.l1tsimglobalsummary)
    return process

def L1TAddInfoOutput(process):
    process.MessageLogger = cms.Service(
        "MessageLogger",
        destinations = cms.untracked.vstring('cout','cerr'),
        cout = cms.untracked.PSet(threshold = cms.untracked.string('INFO')),
        cerr = cms.untracked.PSet(threshold  = cms.untracked.string('WARNING')),
        )
    return process


def L1TAddDebugOutput(process):
    print("L1T INFO:  sending debugging ouput to file l1tdebug.log")
    print("L1T INFO:  add <flags CXXFLAGS=\"-g -D=EDM_ML_DEBUG\"/> in BuildFile.xml of any package you want to debug...")
    process.MessageLogger = cms.Service(
        "MessageLogger",
        destinations = cms.untracked.vstring('l1tdebug','cerr'),
        l1tdebug = cms.untracked.PSet(threshold = cms.untracked.string('DEBUG')),
        #debugModules = cms.untracked.vstring('caloStage1Digis'))
        cerr = cms.untracked.PSet(threshold  = cms.untracked.string('WARNING')),
        debugModules = cms.untracked.vstring('*'))
    return process

def L1TDumpEventData(process):
    print("L1T INFO:  adding EventContentAnalyzer to process schedule")
    process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
    process.l1tdumpevent = cms.Path(process.dumpED)
    process.schedule.append(process.l1tdumpevent)
    return process

def L1TDumpEventSummary(process):
    process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")
    process.l1tdumpeventsetup = cms.Path(process.dumpES)
    process.schedule.append(process.l1tdumpeventsetup)
    return process

def L1TStage2ComparisonRAWvsEMU(process):
    print("L1T INFO:  will dump a comparison of unpacked vs emulated Stage2 content to screen.")
    process.load('L1Trigger.L1TCommon.l1tComparisonStage2RAWvsEMU_cfi')
    process.l1tstage2comparison = cms.Path(process.l1tComparisonStage2RAWvsEMU)
    process.schedule.append(process.l1tstage2comparison)
    return process


def L1TGtStage2ComparisonRAWvsEMU(process):
    print("L1T INFO:  will dump a comparison of unpacked vs emulated GT Stage2 content to screen.")
    process.load('L1Trigger.L1TCommon.l1tComparisonGtStage2RAWvsEMU_cfi')
    process.l1tgtstage2comparison = cms.Path(process.l1tComparisonGtStage2RAWvsEMU)
    process.schedule.append(process.l1tgtstage2comparison)
    return process

def L1TStage2SetPrefireVetoBit(process):
    process.load('L1Trigger.L1TGlobal.simGtExtFakeProd_cfi')
    process.simGtExtFakeProd.tcdsRecordLabel = cms.InputTag("tcdsDigis","tcdsRecord")
    process.l1tstage2gtext = cms.Path(process.simGtExtFakeProd)
    process.schedule.insert(0,process.l1tstage2gtext)
    return process
