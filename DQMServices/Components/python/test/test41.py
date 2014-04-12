import FWCore.ParameterSet.Config as cms

def customise(process):

    # Force a central reset of all MEs in DQMStore at every Run Boundary.
    process.DQMStore.forceResetOnBeginRun = cms.untracked.bool(True)

    # Remove offending L1 sequence.
    process.dqmoffline_step.remove(process.l1TriggerDqmOffline)

    # Save standard DQM Files also after the first step to check if
    # the disagreement is already present at this level or if it
    # coming only from the harvesting step.
    process.load("DQMServices.Components.DQMFileSaver_cfi")
    process.dqmSaver.workflow = cms.untracked.string(\
        "/Jet/Run2011A-BoundaryTest-v1-FirstStep/DQM")
    process.saveROOTFileAtEndRun = cms.Path(process.dqmSaver)
    where = 0
    if getattr(process, 'endjob_step', None):
        where = process.schedule.index(process.endjob_step)
    elif getattr(process, 'DQMoutput_step', None):
        where = process.schedule.index(process.DQMoutput_step)
    process.schedule.insert(where, process.saveROOTFileAtEndRun)

    return(process)
