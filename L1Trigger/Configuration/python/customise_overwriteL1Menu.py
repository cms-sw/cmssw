#
# replace the L1 menu from the global tag with another menu
# see options in L1Trigger_custom.py
#
# V.M. Ghete 2010-06-09

import FWCore.ParameterSet.Config as cms

def customise(process):
    from L1Trigger.Configuration.L1Trigger_custom import customiseL1Menu
    process=customiseL1Menu(process)
    
    return (process)
