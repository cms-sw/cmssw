import FWCore.ParameterSet.Config as cms

def customise(process):

    # Save ByLumi certification flags so that we can also check them
    # using RelMon.
    process.dqmSaver.saveByLumiSection = cms.untracked.int32(1)
    process.dqmSaver.workflow = cms.untracked.string('/Jet/Run2011A-BoundaryTest-v1/DQM')

    # Remove offending L1 sequence.
    process.DQMOffline_SecondStep_PrePOG.remove(process.l1TriggerDqmOfflineClient)

    return(process)
