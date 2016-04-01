import FWCore.ParameterSet.Config as cms

#
# reusable functions
def producers_by_type(process, *types):
    return (module for module in process._Process__producers.values() if module._TypedParameterizable__type in types)

def esproducers_by_type(process, *types):
    return (module for module in process._Process__esproducers.values() if module._TypedParameterizable__type in types)

#
# one action function per PR - put the PR number into the name of the function

# example:
# def customiseFor12718(process):
#     for pset in process._Process__psets.values():
#         if hasattr(pset,'ComponentType'):
#             if (pset.ComponentType == 'CkfBaseTrajectoryFilter'):
#                 if not hasattr(pset,'minGoodStripCharge'):
#                     pset.minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutNone'))
#     return process

# Add quadruplet-specific pixel track duplicate cleaning mode (PR #13753)
def customiseFor13753(process):
    for producer in producers_by_type(process, "PixelTrackProducer"):
        if producer.CleanerPSet.ComponentName.value() == "PixelTrackCleanerBySharedHits" and not hasattr(producer.CleanerPSet, "useQuadrupletAlgo"):
            producer.CleanerPSet.useQuadrupletAlgo = cms.bool(False)
    return process

#
# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):
    import os
    cmsswVersion = os.environ['CMSSW_VERSION']

    if cmsswVersion >= "CMSSW_8_1":
        process = customiseFor13753(process)

    if cmsswVersion >= "CMSSW_8_0":
#       process = customiseFor12718(process)
        pass

    return process
