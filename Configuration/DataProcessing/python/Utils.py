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

