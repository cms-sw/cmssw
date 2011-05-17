#import FWCore.ParameterSet.Config as cms
import os

'''

Helper functions for modifying the tau sequences.


Author: Evan K. Friis, UC Davis

'''

def cmssw_version():
    version_str = os.environ['CMSSW_VERSION'].replace('CMSSW_', '').split('_')
    return (int(version_str[0]), int(version_str[1]), int(version_str[2]))
