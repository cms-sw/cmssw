import FWCore.ParameterSet.Config as cms

# import a full HLT menu
import sys, os
sys.path.append( '%s/src/HLTrigger/Configuration/test' % os.environ['CMSSW_BASE'] )
sys.path.append( '%s/src/HLTrigger/Configuration/test' % os.environ['CMSSW_RELEASE_BASE'] )
from OnData_HLT_GRun import process

# options

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( True )
)

process.source.fileNames = (
    '/store/group/dpg_trigger/comm_trigger/TriggerStudiesGroup/Timing/sample.root',
)

process.maxEvents.input = -1

# load and replace the FastTimerService
if process.FastTimerService:
  del process.FastTimerService

process.load('HLTrigger/Timer/FastTimerService_cff')
process.FastTimerService.printRunSummary          = True
process.FastTimerService.printJobSummary          = True
process.FastTimerService.enableDQM                = True
process.FastTimerService.enableDQMbyModule        = True
process.FastTimerService.enableDQMbyLumiSection   = True
process.FastTimerService.enableDQMbyProcesses     = True
