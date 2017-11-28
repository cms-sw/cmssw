import FWCore.ParameterSet.Config as cms

# helper fuctions
from HLTrigger.Configuration.common import *

# add one customisation function per PR
# - put the PR number into the name of the function
# - add a short comment
# for example:

# CCCTF tuning
# def customiseFor12718(process):
#     for pset in process._Process__psets.values():
#         if hasattr(pset,'ComponentType'):
#             if (pset.ComponentType == 'CkfBaseTrajectoryFilter'):
#                 if not hasattr(pset,'minGoodStripCharge'):
#                     pset.minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutNone'))
#     return process

# Migrate uGT non-CondDB parameters to new cff: remove StableParameters dependence in favour of GlobalParameters
def customiseFor20689(process):
    if hasattr(process,'StableParametersRcdSource'):
        delattr(process,'StableParametersRcdSource')
    if hasattr(process,'StableParameters'):
        delattr(process,'StableParameters')
    if not hasattr(process,'GlobalParameters'):
        from L1Trigger.L1TGlobal.GlobalParameters_cff import GlobalParameters
        process.GlobalParameters = GlobalParameters
    return process

# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):

    # add call to action function in proper order: newest last!
    # process = customiseFor12718(process)

    process = customiseFor20689(process)

    return process
