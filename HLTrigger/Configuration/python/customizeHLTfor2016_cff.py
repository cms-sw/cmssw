import FWCore.ParameterSet.Config as cms

# customisation functions for the HLT configuration
from HLTrigger.Configuration.common import *

# import the relevant eras from Configuration.Eras.*
from Configuration.Eras.Modifier_HLT_2016_cff import HLT_2016


# modify the HLT configuration for the 2016 configuration
def customizeHLTfor2016(process):
    pass


# attach `customizeHLTfor2016' to the `HLT_2016' modifier
def modifyHLTfor2016(process):
    HLT_2016.toModify(process, customizeHLTfor2016)

