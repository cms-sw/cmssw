import FWCore.ParameterSet.Config as cms

import os

##############################################################################
# customisations for L1T demos
#
# Add demonstration modules to cmsDriver customs.
#
##############################################################################

def L1TBasicDemo(process):
    print "L1T INFO:  adding basic demo module to the process."
    process.load('L1Trigger.L1TCommon.l1tBasicDemo_cfi')
    process.l1tBasicDemoPath = cms.Path(process.l1tBasicDemo)
    process.schedule.append(process.l1tBasicDemoPath)
    return process

