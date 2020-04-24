import FWCore.ParameterSet.Config as cms

# instrument the process with the FastTimerService
from HLTrigger.Timer.FastTimerService_cfi import FastTimerService

# print a text summary at the end of the job
FastTimerService.printEventSummary          = False
FastTimerService.printRunSummary            = False
FastTimerService.printJobSummary            = True

# enable per-event DQM plots
FastTimerService.enableDQM                  = True

# disable per-module DQM plots
FastTimerService.enableDQMbyModule          = False

# enable per-event DQM plots by lumisection
FastTimerService.enableDQMbyLumiSection     = True
FastTimerService.dqmLumiSectionsRange       = 2500    # lumisections (23.31 s)

# set the time resolution of the DQM plots
FastTimerService.dqmTimeRange               = 10000.  # ms
FastTimerService.dqmTimeResolution          =    10.  # ms
FastTimerService.dqmPathTimeRange           = 10000.  # ms
FastTimerService.dqmPathTimeResolution      =    10.  # ms
FastTimerService.dqmModuleTimeRange         =   100.  # ms
FastTimerService.dqmModuleTimeResolution    =     0.5 # ms

# set the base DQM folder for the plots
FastTimerService.dqmPath                    = "DQM/TimerService"
FastTimerService.enableDQMbyProcesses       = False
