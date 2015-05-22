#!/usr/bin/env python

import os, re

def isCrabFile(fname):
    """Does this file name match the name convention used by Crab?"""
    base, _ = os.path.splitext(os.path.basename(fname))
    return re.match(".*_\d+_\d+_\w+$", base) is not None

def isBatchFile(fname):
    """Does the file name match the name convention used by cmsBatch?"""
    base, _ = os.path.splitext(os.path.basename(fname))
    return re.match(".*_\d+$", base) is not None

def getFileRegExp(fname, match = True):
    result = None
    if isCrabFile(fname):
        result = '_\d+_\d+_\w+$'
    elif isBatchFile(fname):
        result = "_\d+$"
    if match:
        result = '.*%s' % result
    return result

def getFileGroup(fname):
    """Return the non auto-generated part of the name"""
    
    regexp = getFileRegExp(fname, match = False)
    
    base, _ = os.path.splitext(os.path.basename(fname))
    if regexp is not None:
        tokens = re.split(regexp,base)
        if tokens:
            return tokens[0]
    return None


if __name__ == '__main__':

    crab = 'cmgTuple_10_1_evs.root'
    batch = 'patTuple_97.root'

    assert isCrabFile(crab)
    assert not isCrabFile(batch)

    assert isBatchFile(batch)
    assert not isBatchFile(crab)
    

    assert getFileGroup(crab) == 'cmgTuple'
    assert getFileGroup(batch) == 'patTuple'
    assert getFileGroup('cmg_Tuple_10_1_evs.root') == 'cmg_Tuple'
    assert getFileGroup('pat_Tuple_97.root') == 'pat_Tuple'
