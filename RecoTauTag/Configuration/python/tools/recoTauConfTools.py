#import FWCore.ParameterSet.Config as cms
import os

'''

Helper functions for modifying the tau sequences.


Author: Evan K. Friis, UC Davis

'''

def represents_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

def cmssw_version():
    version_str = os.environ['CMSSW_VERSION'].replace('CMSSW_', '').split('_')
    major_version = int(version_str[0])
    minor_version = int(version_str[1])
    subminor_version = None
    # Correctly deal with IB releases, where the subminor version is X. Return it 
    # as a string, which is always larger than integers.
    if represents_int(version_str[2]):
        subminor_version = int(version_str[2])
    else:
        subminor_version = version_str[2]
    return (version_str[0], version_str[1], version_str[2])
