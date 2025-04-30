import os
import FWCore.ParameterSet.Config as cms

##
## Setup command line options
##
import FWCore.ParameterSet.VarParsing as VarParsing
import sys
options = VarParsing.VarParsing ('standard')
options.register('iteration', 0, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Iteration number")
options.register('isBaseline', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, "Set baseline")
options.register('workingArea', None, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Working folder")
options.register('measName', None, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Folder in which to store results")
options.register('baselineName', "Design", VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Folder in which baseline-trees are found or stored")

# get and parse the command line arguments
options.parseArguments()

##
## Process definition
##
process = cms.Process("ApeEstimatorSummary")

##
## Message Logger
##
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.CalculateAPE=dict()
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
                    firstRun = cms.untracked.uint32(1)
                    )

##
## Number of Events
##
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

### To get default APEs from GT
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
from CondCore.CondDB.CondDB_cfi import *

# does not really matter here because we dont use anything from conditions anyway
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2018_design', '')
# except that we could load an APE tag here, which would then be read out and written to allData_defaultApe.root, but there is really no point

##
## ApeEstimatorSummary
##
from Alignment.APEEstimation.ApeEstimatorSummary_cff import *
process.ApeEstimatorSummarySequence = cms.Sequence()
if options.isBaseline:
  process.ApeEstimatorSummary1 = ApeEstimatorSummaryBaseline.clone(
    # baseline will be set
    BaselineFile = os.path.join(options.workingArea,options.baselineName, "baseline",'allData_baselineApe.root'),
    DefaultFile = os.path.join(options.workingArea,options.baselineName, "baseline", 'allData_defaultApe.root'),
    InputFile = os.path.join(options.workingArea,options.baselineName, "baseline",'allData.root'),
    ResultsFile = os.path.join(options.workingArea,options.baselineName, "baseline",'allData_resultsFile.root'),
  )
  process.ApeEstimatorSummary2 = ApeEstimatorSummaryIter.clone(
    BaselineFile = os.path.join(options.workingArea,options.baselineName, "baseline",'allData_baselineApe.root'),
    InputFile = os.path.join(options.workingArea,options.baselineName, "baseline",'allData.root'),
    ResultsFile = os.path.join(options.workingArea,options.baselineName, "baseline",'allData_resultsFile.root'),
    # files are not in use in baseline mode
    IterationFile = os.path.join(options.workingArea,options.baselineName, "baseline", 'allData_iterationApe.root'),
    DefaultFile = os.path.join(options.workingArea,options.baselineName, "baseline", 'allData_defaultApe.root'),
    ApeOutputFile = os.path.join(options.workingArea,options.baselineName, "baseline", 'allData_apeOutput.txt'),
  )
  process.ApeEstimatorSummarySequence *= process.ApeEstimatorSummary1
  process.ApeEstimatorSummarySequence *= process.ApeEstimatorSummary2
else:
  process.ApeEstimatorSummary1 = ApeEstimatorSummaryIter.clone(
    # keep the same for all jobs
    BaselineFile = os.path.join(options.workingArea,options.baselineName, "baseline",'allData_baselineApe.root'),
    # keep the first one on misaligned geometry for iterations on same geometry (or better use copy of it)
    IterationFile = os.path.join(options.workingArea,options.measName, 'iter'+str(options.iteration), 'allData_iterationApe.root'),
    # change iteration number for these
    InputFile = os.path.join(options.workingArea,options.measName, 'iter'+str(options.iteration), 'allData.root'),
    ResultsFile = os.path.join(options.workingArea,options.measName, 'iter'+str(options.iteration), 'allData_resultsFile.root'),
    ApeOutputFile = os.path.join(options.workingArea,options.measName, 'iter'+str(options.iteration), 'allData_apeOutput.txt'),
    DefaultFile = os.path.join(options.workingArea,options.measName, 'iter'+str(options.iteration), 'allData_defaultApe.root'),
  )
  process.ApeEstimatorSummarySequence *= process.ApeEstimatorSummary1



##
## Path
##
process.p = cms.Path(
    process.ApeEstimatorSummarySequence
)






-- dummy change --
