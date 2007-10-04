#!/usr/bin/env python

import FWCore.ParameterSet.Config as cms
import sys



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





##########################
if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "usage: testSkimConfig <filename>"

    print "Checking skim config file", sys.argv[1]
    process = cms.include(sys.argv[1])

    for outputModuleName in process._Process__outputmodules:
        outputModule = getattr(process, outputModuleName)
        checkOutputModuleConfig(outputModule)
