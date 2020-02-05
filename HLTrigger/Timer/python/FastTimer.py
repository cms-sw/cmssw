import FWCore.ParameterSet.Config as cms

def addService(process, multirun=False):
    # remove any instance of the FastTimerService
    if 'FastTimerService' in process.__dict__:
        del process.FastTimerService

    # instrument the menu with the FastTimerService
    process.load("HLTrigger.Timer.FastTimerService_cfi")

    # print a text summary at the end of the job
    process.FastTimerService.printEventSummary        = False
    process.FastTimerService.printRunSummary          = False
    process.FastTimerService.printJobSummary          = True

    # enable DQM plots
    process.FastTimerService.enableDQM                = True

    # enable per-path DQM plots (starting with CMSSW 9.2.3-patch2)
    process.FastTimerService.enableDQMbyPath          = True

    # enable per-module DQM plots
    process.FastTimerService.enableDQMbyModule        = True

    # enable DQM plots vs lumisection
    process.FastTimerService.enableDQMbyLumiSection   = True
    process.FastTimerService.dqmLumiSectionsRange     = 2500    # lumisections (23.31 s)

    # set the time resolution of the DQM plots
    process.FastTimerService.dqmTimeRange             = 1000.   # ms
    process.FastTimerService.dqmTimeResolution        =    5.   # ms
    process.FastTimerService.dqmPathTimeRange         =  100.   # ms
    process.FastTimerService.dqmPathTimeResolution    =    0.5  # ms
    process.FastTimerService.dqmModuleTimeRange       =   40.   # ms
    process.FastTimerService.dqmModuleTimeResolution  =    0.2  # ms

    # set the base DQM folder for the plots
    process.FastTimerService.dqmPath                  = "HLT/TimerService"
    process.FastTimerService.enableDQMbyProcesses     = False

    if multirun:
        # disable the per-lumisection plots
        process.FastTimerService.enableDQMbyLumiSection = False

    return process

def addOutput(process):
    # save the DQM plots in the DQMIO format
    process.dqmOutput = cms.OutputModule("DQMRootOutputModule",
        fileName = cms.untracked.string("DQM.root")
    )
    process.FastTimerOutput = cms.EndPath(process.dqmOutput)
    process.schedule.append(process.FastTimerOutput)

    return process

def addPrint(process):
    # enable text dump
    if not hasattr(process,'MessageLogger'):
        process.load('FWCore.MessageService.MessageLogger_cfi')
    process.MessageLogger.categories.append('FastReport')
    process.MessageLogger.cerr.FastReport = cms.untracked.PSet( limit = cms.untracked.int32( 10000000 ) )
    return process

def addHarvest(process):
    # DQMStore service
    if not hasattr(process,'DQMStore'):
        process.load('DQMServices.Core.DQMStore_cfi')

    # FastTimerService client
    process.load('HLTrigger.Timer.fastTimerServiceClient_cfi')
    process.fastTimerServiceClient.dqmPath = "HLT/TimerService"

    # DQM file saver
    process.load('DQMServices.Components.DQMFileSaver_cfi')
    process.dqmSaver.workflow = "/HLT/FastTimerService/All"

    process.DQMFileSaverOutput = cms.EndPath(process.fastTimerServiceClient + process.dqmSaver)
    process.schedule.append(process.DQMFileSaverOutput)

    return process

# customise functions for cmsDriver

def customise_timer_service(process):
	process = addService(process)
	process = addOutput(process)
	return process

def customise_timer_service_singlejob(process):
	process = addService(process)
	process = addHarvest(process)
	return process

def customise_timer_service_multirun(process):
	process = addService(process,True)
	process = addOutput(process)
	return process

def customise_timer_service_print(process):
	process = addService(process)
	process = addPrint(process)
	return process

