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

def customiseForXXXXX(process):
    for pset in process._Process__psets.values():
        if hasattr(pset,'ComponentType'):
            if (pset.ComponentType == 'CkfBaseTrajectoryFilter'):
                value = cms.int32(13)
                if hasattr(pset,'minNumberOfHits'):
                    value = getattr(pset,'minNumberOfHits')
                    delattr(pset,'minNumberOfHits')
                if not hasattr(pset,'minNumberOfHitsForLoopers'):
                    pset.minNumberOfHitsForLoopers = value
                if not hasattr(pset,'minNumberOfHitsPerLoop'):
                    pset.minNumberOfHitsPerLoop = cms.int32(4)
                if not hasattr(pset,'extraNumberOfHitsBeforeTheFirstLoop'):
                    pset.extraNumberOfHitsBeforeTheFirstLoop = cms.int32(4)
                if not hasattr(pset,'maxLostHitsFraction'):
                    pset.maxLostHitsFraction = cms.double(999.)
                if not hasattr(pset,'constantValueForLostHitsFractionFilter'):
                    pset.constantValueForLostHitsFractionFilter = cms.double(1.)
                if not hasattr(pset,'minimumNumberOfHits'):
                    pset.minimumNumberOfHits = cms.int32(5)
                if not hasattr(pset,'seedPairPenalty'):
                    pset.seedPairPenalty = cms.int32(0)
    return process

#
# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):
    import os
    cmsswVersion = os.environ['CMSSW_VERSION']

    if cmsswVersion >= "CMSSW_8_0":
#       process = customiseFor12718(process)
        process = customiseForXXXXX(process)
        pass

    return process
