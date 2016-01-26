import FWCore.ParameterSet.Config as cms

#
# reusable functions
def producers_by_type(process, *types):
    return (module for module in process._Process__producers.values() if module._TypedParameterizable__type in types)

def esproducers_by_type(process, *types):
    return (module for module in process._Process__esproducers.values() if module._TypedParameterizable__type in types)

#
# one action function per PR - put the PR number into the name of the function

def customiseFor12718(process):
    for pset in process._Process__psets.values():
        if hasattr(pset,'ComponentType'):
            if (pset.ComponentType == 'CkfBaseTrajectoryFilter'):
                if not hasattr(pset,'minGoodStripCharge'):
                    pset.minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutNone'))
    return process

#
# CMSSW version specific customizations
def customiseHLTforCMSSW(process, menuType="GRun", fastSim=False):
    import os
    cmsswVersion = os.environ['CMSSW_VERSION']

    if cmsswVersion >= "CMSSW_8_0":
        process = customiseFor12718(process)

    return process
