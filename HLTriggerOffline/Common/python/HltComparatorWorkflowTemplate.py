# Original Author: James Jackson
# $Id: HltComparatorWorkflowTemplate.py,v 1.2 2009/07/19 14:34:19 wittich Exp $
import FWCore.ParameterSet.Config as cms

process = cms.Process("HltRerun")

# summary
process.options = cms.untracked.PSet( 
    wantSummary = cms.untracked.bool(True) 
) 


# Logging
process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.threshold = 'ERROR'

# TFileService
process.TFileService = cms.Service("TFileService", fileName = cms.string("histograms.root"))

#process.dump = cms.EDAnalyzer('EventContentAnalyzer')


# Data to run. You'll want to change this.
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/365/664D66FA-01AB-DD11-B9E5-001617C3B76A.root'
    ),
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Trigger analysis - online
process.hltTrigReport = cms.EDAnalyzer( "HLTrigReport",
    HLTriggerResults = cms.InputTag( 'TriggerResults','','HLT' )
)

# Trigger analysis - offline
process.hltTrigReportRerun = cms.EDAnalyzer( "HLTrigReport",
    HLTriggerResults = cms.InputTag( 'TriggerResults','','HltRerun' )
)

process.load("HLTriggerOffline.Common.HltComparator_cfi")



### output 
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('mismatchedTriggers.root'),
         outputCommands = cms.untracked.vstring('drop *',
                                                "keep *_*_*_HltRerun"
                                                )
)


# Final reporting and output. The output is only for discrepant events.
process.HLTAnalysisEndPath = cms.EndPath( process.hltTrigReport + process.hltTrigReportRerun+process.hltComparator+process.out)

# load HLT configuration -- something like this will be automatically appended.
#process.load("PwTest.HltTester.JobHLTConfig_13691_32003_1244677280_cff")
