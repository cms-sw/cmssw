#!/usr/bin/env python
import FWCore.ParameterSet.Config as cms
import sys
import FWCore.ParameterSet.SequenceTypes as sqt
import FWCore.ParameterSet.Modules as mod


def checkOutputModuleConfig(module):
    """Check if a PoolOutputModule is properly configured"""
    if hasattr(module,"dataset"):
        dataset = getattr(module,"dataset")
        try:
            dataTier = getattr(dataset,"dataTier")
            filterName = getattr(dataset,"filterName")
        except:
            print "Module", module, "has a malformed PSet dataset"
    else:
        print "Module", module, "has no PSet dataset defined"


def getModulesFromSequence(sequence,list):
    item = sequence._seq
    if isinstance(item,mod._Module):
        list.append(item)
    elif isinstance(item,cms.Sequence):
         getModulesFromSequence(item,list)
    else:
         _getModulesFromOp(item,list)
                                                    

def _getModulesFromOp(op,list):
    for item in dir(op):
        o = getattr(op,item)
        if isinstance(o,mod._Module):
            list.append(o)
        elif isinstance(o, cms.Sequence):
            _getModulesFromOp(o,list)
        elif isinstance(o,sqt._Sequenceable):
            _getModulesFromOp(o,list)
                    

def extractUsedOutputs(process):
    allEndPathModules = []
    for name in process._Process__endpaths:
        endpath = getattr(process,name)
        list = []
        getModulesFromSequence(endpath,list)
        allEndPathModules.extend(list)
    allUsedOutputModules = []
    for module in allEndPathModules:
        if isinstance(module, cms.OutputModule):
            allUsedOutputModules.append(module)
    return allUsedOutputModules



##########################
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print "usage: testSkimConfig <filenames>"

    for scriptName in sys.argv[1:]:      
        print "Checking skim config file", scriptName

        process = cms.include(scriptName)
        #print "+ python parseable"
    
        print "checking", len (process._Process__outputmodules), "output modules"
        for outputModuleName in process._Process__outputmodules:
            print "  ", outputModuleName
            outputModule = getattr(process, outputModuleName)
            checkOutputModuleConfig(outputModule)

        usedOutputs = extractUsedOutputs(process)
        print "Actually used: ", len(usedOutputs)
        for module in usedOutputs:
            print "  ", module.label()
