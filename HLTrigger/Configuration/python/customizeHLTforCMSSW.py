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

# Add new parameters to RecoTrackRefSelector
def customiseFor19029(process):
    for producer in producers_by_type(process, "RecoTrackRefSelector"):
        if not hasattr(producer, "minPhi"):
            producer.minPhi = cms.double(-3.2)
            producer.maxPhi = cms.double(3.2)
    return process

def customiseFor20269(process) :
    for producer in esproducers_by_type(process, "ClusterShapeHitFilterESProducer"):
         producer.PixelShapeFile   = cms.string('RecoPixelVertexing/PixelLowPtUtilities/data/pixelShapePhase1_noL1.par')
         producer.PixelShapeFileL1 = cms.string('RecoPixelVertexing/PixelLowPtUtilities/data/pixelShapePhase1_loose.par')
    return process

# Migrate uGT non-CondDB parameters to new cff: remove StableParameters dependence in favour of GlobalParameters
def customiseFor19989(process):
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

    process = customiseFor19029(process)
    process = customiseFor20269(process)
    process = customiseFor19989(process)

    return process
