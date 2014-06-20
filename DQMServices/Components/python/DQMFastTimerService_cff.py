import FWCore.ParameterSet.Config as cms

# instrument the process with the FastTimerService
from HLTrigger.Timer.FastTimerService_cfi import FastTimerService

# this is currently ignored in 7.x, and always uses the real time clock
FastTimerService.useRealTimeClock           = False

# enable specific features
FastTimerService.enableTimingPaths          = True
FastTimerService.enableTimingModules        = False
FastTimerService.enableTimingExclusive      = False

# print a text summary at the end of the job
FastTimerService.enableTimingSummary        = True

# skip the first path (useful for HLT timing studies to disregard the time spent loading event and conditions data)
FastTimerService.skipFirstPath              = False

# enable per-event DQM plots
FastTimerService.enableDQM                  = True

# enable per-path DQM plots
FastTimerService.enableDQMbyPathActive      = True
FastTimerService.enableDQMbyPathTotal       = True
FastTimerService.enableDQMbyPathOverhead    = False
FastTimerService.enableDQMbyPathDetails     = True
FastTimerService.enableDQMbyPathCounters    = True
FastTimerService.enableDQMbyPathExclusive   = False

# enable per-module DQM plots
FastTimerService.enableDQMbyModule          = False
FastTimerService.enableDQMbyModuleType      = False

# enable per-event DQM sumary plots
FastTimerService.enableDQMSummary           = True

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
