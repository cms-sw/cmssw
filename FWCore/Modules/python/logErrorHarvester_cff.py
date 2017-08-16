import FWCore.ParameterSet.Config as cms
from FWCore.Modules.logErrorHarvester_cfi import logErrorHarvester

def customiseLogErrorHarvesterUsingOutputCommands(process):
    if not hasattr(process,'logErrorHarvester'):
        return process

    modulesFromOutput = set()
    for o in process.outputModules_().itervalues():
        if not hasattr(o,"outputCommands"):
            continue
        for ln in o.outputCommands.value():
            if -1 != ln.find("keep"):
                s = ln.split("_")
                if len(s)>1:
                    if s[1].find("*")==-1 and s[1] != 'logErrorHarvester':
                        modulesFromOutput.add(s[1])
    process.logErrorHarvester.includeModules =cms.untracked.vstring(*modulesFromOutput)
    return process
