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

# Matching ECAL selective readout in particle flow, need a new input with online Selective Readout Flags
def customiseFor17794(process):
     for edproducer in producers_by_type(process, "PFRecHitProducer"):
          if hasattr(edproducer,'producers'):
               for pset in edproducer.producers:
                    if (pset.name == 'PFEBRecHitCreator' or pset.name == 'PFEERecHitCreator'):
                         if not hasattr(pset,'srFlags'):
                              pset.srFlags = cms.InputTag('hltEcalDigis')
     return process


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

# DA clusterizer tuning
def customiseFor18330(process):
     for producer in producers_by_type(process, "PrimaryVertexProducer"):
          if producer.TkFilterParameters.algorithm.value() == "filter" and not hasattr(producer.TkFilterParameters, "maxEta"):
               producer.TkFilterParameters.maxEta = cms.double(100.0)

          if producer.TkClusParameters.algorithm.value() == "DA_vect":
               if abs(producer.TkClusParameters.TkDAClusParameters.Tmin.value() - 4.0) < 1e-3:
                    # default value was changed, going from 4.0 -> 2.4 should give no change in results
                    producer.TkClusParameters.TkDAClusParameters.Tmin = 2.4
               if not hasattr(producer.TkClusParameters.TkDAClusParameters, "Tpurge"):
                    producer.TkClusParameters.TkDAClusParameters.Tpurge = cms.double(2.0)
               if not hasattr(producer.TkClusParameters.TkDAClusParameters, "Tstop"):
                    producer.TkClusParameters.TkDAClusParameters.Tstop = cms.double(0.5)
               if not hasattr(producer.TkClusParameters.TkDAClusParameters, "zmerge"):
                    producer.TkClusParameters.TkDAClusParameters.zmerge = cms.double(1e-2)
               if not hasattr(producer.TkClusParameters.TkDAClusParameters, "uniquetrkweight"):
                    producer.TkClusParameters.TkDAClusParameters.uniquetrkweight = cms.double(0.9)

          for pset in producer.vertexCollections:
               if pset.algorithm.value() == "AdaptiveVertexFitter" and not hasattr(pset, "chi2cutoff"):
                    pset.chi2cutoff = cms.double(3.0)

     return process

# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):
    # add call to action function in proper order: newest last!
    # process = customiseFor12718(process)
    process = customiseFor17771(process)
    process = customiseFor17792(process)
    process = customiseFor17794(process)
    process = customiseFor18330(process)

    return process
