import os

import FWCore.ParameterSet.Config as cms



##
## Setup command line options
##
import FWCore.ParameterSet.VarParsing as VarParsing
import sys
options = VarParsing.VarParsing ('standard')
options.register('iterNumber', 0, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Iteration number")
options.register('setBaseline', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, "Set baseline")



# get and parse the command line arguments
if( hasattr(sys, "argv") ):
    for args in sys.argv :
        arg = args.split(',')
        for val in arg:
            val = val.split('=')
            if(len(val)==2):
                setattr(options,val[0], val[1])

print "Iteration number: ", options.iterNumber
print "Set baseline: ", options.setBaseline



##
## Process definition
##
process = cms.Process("ApeEstimatorSummary")



##
## Message Logger
##
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.categories.append('CalculateAPE')
process.MessageLogger.cerr.INFO.limit = 0
process.MessageLogger.cerr.default.limit = 0
process.MessageLogger.cerr.CalculateAPE = cms.untracked.PSet(limit = cms.untracked.int32(-1))
process.MessageLogger.cerr.FwkReport.reportEvery = 1000 ## really show only every 1000th



##
## Process options
##
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
)



##
## Input Files
##
process.source = cms.Source("EmptySource",
					numberEventsInRun = cms.untracked.uint32(1),
					firstRun = cms.untracked.uint32(246994)
					)



##
## Number of Events
##
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

### To get default APEs from GT
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
import CalibTracker.Configuration.Common.PoolDBESSource_cfi

process.GlobalTag.globaltag = 'GR_P_V56'

process.myTrackerAlignmentErr = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
  connect = 'frontier://FrontierProd/CMS_CONDITIONS',
  toGet = [
	cms.PSet(
	  record = cms.string('TrackerAlignmentErrorExtendedRcd'),
	  tag = cms.string('TrackerAlignmentExtendedErr_2009_v2_express_IOVs')
	),
  ],
)
process.es_prefer_trackerAlignmentErr = cms.ESPrefer("PoolDBESSource","myTrackerAlignmentErr")


##
## ApeEstimatorSummary
##
from Alignment.APEEstimation.ApeEstimatorSummary_cff import *
process.ApeEstimatorSummarySequence = cms.Sequence()
if options.setBaseline:
  process.ApeEstimatorSummary1 = ApeEstimatorSummaryBaseline.clone(
    # baseline will be set
    BaselineFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/Design/baseline/allData_baselineApe.root',
    DefaultFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/Design/baseline/allData_defaultApe.root',
    InputFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/Design/baseline/allData.root',
    ResultsFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/Design/baseline/allData_resultsFile.root',
  )
  process.ApeEstimatorSummary2 = ApeEstimatorSummaryIter.clone(
    BaselineFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/Design/baseline/allData_baselineApe.root',
    InputFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/Design/baseline/allData.root',
    ResultsFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/Design/baseline/allData_resultsFile.root',
    # files are not in use in baseline mode
    IterationFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/Design/baseline/allData_iterationApe.root',
    DefaultFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/Design/baseline/allData_defaultApe.root',
    ApeOutputFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/Design/baseline/allData_apeOutput.txt',
  )
  process.ApeEstimatorSummarySequence *= process.ApeEstimatorSummary1
  process.ApeEstimatorSummarySequence *= process.ApeEstimatorSummary2
else:
  process.ApeEstimatorSummary1 = ApeEstimatorSummaryIter.clone(
    # keep the same for all jobs
    BaselineFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/Design/baseline/allData_baselineApe.root',
    # keep the first one on misaligned geometry for iterations on same geometry (or better use copy of it)
    IterationFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/workingArea/iter'+str(options.iterNumber)+'/allData_iterationApe.root',
    # change iteration number for these
    InputFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/workingArea/iter'+str(options.iterNumber)+'/allData.root',
    ResultsFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/workingArea/iter'+str(options.iterNumber)+'/allData_resultsFile.root',
    ApeOutputFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/workingArea/iter'+str(options.iterNumber)+'/allData_apeOutput.txt',
    DefaultFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/workingArea/iter'+str(options.iterNumber)+'/allData_defaultApe.root',
  )
  process.ApeEstimatorSummarySequence *= process.ApeEstimatorSummary1



##
## Path
##
process.p = cms.Path(
    process.ApeEstimatorSummarySequence
)






