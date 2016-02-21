import FWCore.ParameterSet.Config as cms

import os

##############################################################################
# customisations for L1T utilities
#
# customisations which add utilities features such as debugging of L1T,
#    summary module, etc.
#
##############################################################################

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
    print "L1T INFO:  will dump a summary of unpacked Stage1 content to screen."
    process.load('L1Trigger.L1TCommon.l1tSummaryStage1Digis_cfi')
    process.l1tstage1summary = cms.Path(process.l1tSummaryStage1Digis)
    process.schedule.append(process.l1tstage1summary)
    return process

def L1TStage2DigisSummary(process):
    print "L1T INFO:  will dump a summary of unpacked Stage2 content to screen."    
    process.load('L1Trigger.L1TCommon.l1tSummaryStage2Digis_cfi')
    process.l1tstage2summary = cms.Path(process.l1tSummaryStage2Digis)
    process.schedule.append(process.l1tstage2summary)
    return process

def L1TStage1SimDigisSummary(process):
    print "L1T INFO:  will dump a summary of simulated Stage1 content to screen."
    process.load('L1Trigger.L1TCommon.l1tSummaryStage1SimDigis_cfi')
    process.l1tsimstage1summary = cms.Path(process.l1tSummaryStage1SimDigis)
    process.schedule.append(process.l1tsimstage1summary)
    return process

def L1TStage2SimDigisSummary(process):
    print "L1T INFO:  will dump a summary of simulated Stage2 content to screen."    
    process.load('L1Trigger.L1TCommon.l1tSummaryStage2SimDigis_cfi')
    process.l1tsimstage2summary = cms.Path(process.l1tSummaryStage2SimDigis)
    process.schedule.append(process.l1tsimstage2summary)
    return process


def L1TAddDebugOutput(process):
    print "L1T INFO:  sending debugging ouput to file l1tdebug.log"
    print "L1T INFO:  add <flags CXXFLAGS=\"-g -D=EDM_ML_DEBUG\"/> in BuildFile.xml of any package you want to debug..."
    process.MessageLogger = cms.Service(
        "MessageLogger",
        destinations = cms.untracked.vstring('l1tdebug','cerr'),                                 
        l1tdebug = cms.untracked.PSet(threshold = cms.untracked.string('DEBUG')),
        #debugModules = cms.untracked.vstring('caloStage1Digis'))
        cerr = cms.untracked.PSet(threshold  = cms.untracked.string('WARNING')),
        debugModules = cms.untracked.vstring('*'))
    return process

def L1TDumpEventData(process):
    print "L1T INFO:  adding EventContentAnalyzer to process schedule"
    process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
    process.l1tdumpevent = cms.Path(process.dumpED)
    process.schedule.append(process.l1tdumpevent)
    return process

def L1TDumpEventSummary(process):
    process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")
    process.l1tdumpeventsetup = cms.Path(process.dumpES)
    process.schedule.append(process.l1tdumpeventsetup)
    return process




