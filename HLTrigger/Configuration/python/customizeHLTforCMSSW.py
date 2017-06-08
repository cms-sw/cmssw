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

# EG filter enhancements PR #18559
def customiseFor18559(process):
    for filt in filters_by_type(process, "HLTEgammaGenericFilter", "HLTMuonGenericFilter", "HLTEgammaGenericQuadraticFilter", "HLTEgammaGenericQuadraticEtaFilter"):
        if not hasattr(filt, "doRhoCorrection"):
            filt.doRhoCorrection = cms.bool( False )
            filt.rhoTag = cms.InputTag( "" )
            filt.effectiveAreas = cms.vdouble( 0.0 )
            filt.absEtaLowEdges = cms.vdouble( 0.0 )
            filt.rhoMax = cms.double( 9.9999999E7 )
            filt.rhoScale = cms.double( 1.0 )

    for filt in filters_by_type(process, "HLTEgammaGenericFilter", "HLTMuonGenericFilter", "HLTEgammaGenericQuadraticFilter"):
        if not hasattr(filt, "energyLowEdges"):
            cutRegularEB = filt.thrRegularEB.value()
            cutRegularEE = filt.thrRegularEE.value()
            cutOverEEB = filt.thrOverEEB.value()
            cutOverEEE = filt.thrOverEEE.value()
            cutOverE2EB = filt.thrOverE2EB.value()
            cutOverE2EE = filt.thrOverE2EE.value()

            del filt.thrRegularEB
            del filt.thrRegularEE
            del filt.thrOverEEB
            del filt.thrOverEEE
            del filt.thrOverE2EB
            del filt.thrOverE2EE

            filt.energyLowEdges = cms.vdouble( 0.0 )
            filt.thrRegularEB = cms.vdouble( cutRegularEB )
            filt.thrRegularEE = cms.vdouble( cutRegularEE )
            filt.thrOverEEB = cms.vdouble( cutOverEEB )
            filt.thrOverEEE = cms.vdouble( cutOverEEE )
            filt.thrOverE2EB = cms.vdouble( cutOverE2EB )
            filt.thrOverE2EE = cms.vdouble( cutOverE2EE )

    for filt in filters_by_type(process, "HLTEgammaGenericQuadraticEtaFilter"):
        if not hasattr(filt, "energyLowEdges"):
            cutRegularEB1 = filt.thrRegularEB1.value()
            cutRegularEE1 = filt.thrRegularEE1.value()
            cutOverEEB1 = filt.thrOverEEB1.value()
            cutOverEEE1 = filt.thrOverEEE1.value()
            cutOverE2EB1 = filt.thrOverE2EB1.value()
            cutOverE2EE1 = filt.thrOverE2EE1.value()
            cutRegularEB2 = filt.thrRegularEB2.value()
            cutRegularEE2 = filt.thrRegularEE2.value()
            cutOverEEB2 = filt.thrOverEEB2.value()
            cutOverEEE2 = filt.thrOverEEE2.value()
            cutOverE2EB2 = filt.thrOverE2EB2.value()
            cutOverE2EE2 = filt.thrOverE2EE2.value()

            del filt.thrRegularEB1
            del filt.thrRegularEE1
            del filt.thrOverEEB1
            del filt.thrOverEEE1
            del filt.thrOverE2EB1
            del filt.thrOverE2EE1
            del filt.thrRegularEB2
            del filt.thrRegularEE2
            del filt.thrOverEEB2
            del filt.thrOverEEE2
            del filt.thrOverE2EB2
            del filt.thrOverE2EE2

            filt.energyLowEdges = cms.vdouble( 0.0 )
            filt.thrRegularEB1 = cms.vdouble( cutRegularEB1 )
            filt.thrRegularEE1 = cms.vdouble( cutRegularEE1 )
            filt.thrOverEEB1 = cms.vdouble( cutOverEEB1 )
            filt.thrOverEEE1 = cms.vdouble( cutOverEEE1 )
            filt.thrOverE2EB1 = cms.vdouble( cutOverE2EB1 )
            filt.thrOverE2EE1 = cms.vdouble( cutOverE2EE1 )
            filt.thrRegularEB2 = cms.vdouble( cutRegularEB2 )
            filt.thrRegularEE2 = cms.vdouble( cutRegularEE2 )
            filt.thrOverEEB2 = cms.vdouble( cutOverEEB2 )
            filt.thrOverEEE2 = cms.vdouble( cutOverEEE2 )
            filt.thrOverE2EB2 = cms.vdouble( cutOverE2EB2 )
            filt.thrOverE2EE2 = cms.vdouble( cutOverE2EE2 )
    return process

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

# Add new parameters required by RecoTauBuilderConePlugin
def customiseFor18429(process):
     for producer in producers_by_type(process, "RecoTauProducer"):
          if hasattr(producer,'builders'):
               for pset in producer.builders:
                    if not hasattr(pset,'minAbsPhotonSumPt_insideSignalCone'):
                         pset.minAbsPhotonSumPt_insideSignalCone = cms.double(2.5)
                    if not hasattr(pset,'minRelPhotonSumPt_insideSignalCone'):
                         pset.minRelPhotonSumPt_insideSignalCone = cms.double(0.10)

     return process

# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):

    if (menuType == "GRun2016"):
        # GRun2016 is a 90X menu
        process = customiseFor17771(process)
        process = customiseFor17792(process)
        process = customiseFor17794(process)
        process = customiseFor18330(process)
        process = customiseFor18429(process)
        process = customiseFor18559(process)

    # add call to action function in proper order: newest last!
    # process = customiseFor12718(process)

    return process
