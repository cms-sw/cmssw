import FWCore.ParameterSet.Config as cms

def customise(process):
    # Remove the old RNGState product and Trigger on output
    RNGStateCleaning= cms.PSet(
        outputCommands=cms.untracked.vstring('drop RandomEngineStates_*_*_*',
                                             'keep RandomEngineStates_*_*_'+process.name_())
        )

    for item in process.outputModules_().values():
        item.outputCommands.extend(RNGStateCleaning.outputCommands)

    TRGResultCleaning= cms.PSet(
        outputCommands=cms.untracked.vstring('drop edmTriggerResults_*_*_*',
                                             'keep edmTriggerResults_*_*_'+process.name_())
        )

    for item in process.outputModules_().values():
        item.outputCommands.extend(TRGResultCleaning.outputCommands)


    return(process)
