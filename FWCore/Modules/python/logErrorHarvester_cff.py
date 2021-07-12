import FWCore.ParameterSet.Config as cms
from FWCore.Modules.logErrorHarvester_cfi import logErrorHarvester
import six

def customiseLogErrorHarvesterUsingOutputCommands(process):
    logName = 'logErrorHarvester'
    if not hasattr(process,logName):
        return process

    modulesFromAllOutput = set()
    onlyOneOutput = (len(process.outputModules_()) == 1)
    for o in six.itervalues(process.outputModules_()):
        if not hasattr(o,"outputCommands"):
            continue
        modulesFromOutput = set()
        storesLogs = False
        if (not onlyOneOutput) and hasattr(o,"SelectEvents") and hasattr(o.SelectEvents,"SelectEvents") and o.SelectEvents.SelectEvents:
            #if the output module is skimming events, we do not want to force running of 
            # unscheduled modules
            continue
        for ln in o.outputCommands.value():
            if -1 != ln.find("keep"):
                s = ln.split("_")
                if len(s)>1:
                    if s[1].find("*")==-1:
                        if s[1] != logName:
                            modulesFromOutput.add(s[1])
                        else:
                            storesLogs = True
        if storesLogs:
            modulesFromAllOutput =modulesFromAllOutput.union(modulesFromOutput)
    if hasattr(process.logErrorHarvester,"includeModules"):
        #need to exclude items from includeModules which are not being stored
        includeMods = set(process.logErrorHarvester.includeModules)
        toExclude = includeMods.difference(modulesFromAllOutput)
        if hasattr(process.logErrorHarvester,"excludeModules"):
            toExclude = toExclude.union(set(process.logErrorHarvester.excludeModules.value()))
        process.logErrorHarvester.excludeModules = cms.untracked.vstring(sorted(toExclude))
    else:
        process.logErrorHarvester.includeModules = cms.untracked.vstring(sorted(modulesFromAllOutput))
    return process
