#!/usr/bin/env python
"""
_Utils_

Module containing some utility tools

"""

def stepALCAPRODUCER(skims):
    """
    _stepALCAPRODUCER_

    Creates and returns the configuration string for the ALCAPRODUCER step
    starting from the list of AlcaReco path to be run.

    """

    step = ''
    if len(skims) >0:
        step = ',ALCAPRODUCER:'
        for skim in skims:
            step += (skim+"+")
        step = step.rstrip('+')
    return step

def addMonitoring(process):
    """
    _addMonitoring_
    
    Add the monitoring services to the process provided
    in order to write out performance summaries to the framework job report
    """
    import FWCore.ParameterSet.Config as cms
    
    process.add_(cms.Service("SimpleMemoryCheck"))
    process.add_(cms.Service("Timing"))
    process.Timing.summaryOnly = cms.untracked(cms.bool(True))
    
    return process
