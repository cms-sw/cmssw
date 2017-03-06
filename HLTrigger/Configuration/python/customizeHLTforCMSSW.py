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


# Dynamic track algo priority order
def customiseFor17771(process):
    if not hasattr(process, "hltTrackAlgoPriorityOrder"):
        from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder
        process.hltTrackAlgoPriorityOrder = trackAlgoPriorityOrder.clone(
            ComponentName = "hltTrackAlgoPriorityOrder",
            algoOrder = [] # HLT iteration order is correct in the hard-coded default
        )

    for producer in producers_by_type(process, "SimpleTrackListMerger", "TrackCollectionMerger", "TrackListMerger"):
        if not hasattr(producer, "trackAlgoPriorityOrder"):
            producer.trackAlgoPriorityOrder = cms.string("hltTrackAlgoPriorityOrder")
    return process

# Add optional SeedStopReason to CkfTrackCandidateMaker
def customiseFor17792(process):
    for producer in producers_by_type(process, "CkfTrackCandidateMaker"):
        if not hasattr(producer, "produceSeedStopReasons"):
            producer.produceSeedStopReasons = cms.bool(False)
    return process

# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):
    # add call to action function in proper order: newest last!
    # process = customiseFor12718(process)
    process = customiseFor17771(process)
    process = customiseFor17792(process)

    return process
