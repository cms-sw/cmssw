import FWCore.ParameterSet.Config as cms
import re
import itertools

from FWCore.ParameterSet.MassReplace import massReplaceInputTag
from HeterogeneousCore.AlpakaCore.functions import *
from HLTrigger.Configuration.common import *

## useful functions

def _rename_edmodule(process, oldModuleLabel, newModuleLabel, typeBlackList):
    if not hasattr(process, oldModuleLabel) or hasattr(process, newModuleLabel) or oldModuleLabel == newModuleLabel:
        return process
    oldObj = getattr(process, oldModuleLabel)
    if oldObj.type_() in typeBlackList:
        return process
    setattr(process, newModuleLabel, oldObj.clone())
    newObj = getattr(process, newModuleLabel)
    process = _replace_object(process, oldObj, newObj)
    process.__delattr__(oldModuleLabel)
    process = massReplaceInputTag(process, oldModuleLabel, newModuleLabel, False, True, False)
    for outputModuleLabel in process.outputModules_():
        outputModule = getattr(process, outputModuleLabel)
        if not hasattr(outputModule, 'outputCommands'):
            continue
        for outputCmdIdx, outputCmd in enumerate(outputModule.outputCommands):
            outputModule.outputCommands[outputCmdIdx] = outputCmd.replace(f'_{oldModuleLabel}_', f'_{newModuleLabel}_')
    return process

def _rename_edmodules(process, matchExpr, oldStr, newStr, typeBlackList):
    for moduleDict in [process.producers_(), process.filters_(), process.analyzers_()]:
        moduleLabels = list(moduleDict.keys())
        for moduleLabel in moduleLabels:
            if bool(re.match(matchExpr, moduleLabel)):
                moduleLabelNew = moduleLabel.replace(oldStr, '') + newStr
                process = _rename_edmodule(process, moduleLabel, moduleLabelNew, typeBlackList)
    return process

def _rename_container(process, oldContainerLabel, newContainerLabel):
    if not hasattr(process, oldContainerLabel) or hasattr(process, newContainerLabel) or oldContainerLabel == newContainerLabel:
        return process
    oldObj = getattr(process, oldContainerLabel)
    setattr(process, newContainerLabel, oldObj.copy())
    newObj = getattr(process, newContainerLabel)
    process = _replace_object(process, oldObj, newObj)
    process.__delattr__(oldContainerLabel)
    return process

def _rename_containers(process, matchExpr, oldStr, newStr):
    for containerName in itertools.chain(
        process.sequences_().keys(),
        process.tasks_().keys(),
        process.conditionaltasks_().keys()
    ):
        if bool(re.match(matchExpr, containerName)):
            containerNameNew = containerName.replace(oldStr, '') + newStr
            process = _rename_container(process, containerName, containerNameNew)
    return process



def customizeHLTforAlpaka(process):
    process.load('Configuration.StandardSequences.Accelerators_cff')

    return process
