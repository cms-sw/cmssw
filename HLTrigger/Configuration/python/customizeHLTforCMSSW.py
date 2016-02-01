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

def customiseFor13062(process):
    for module in producers_by_type(process,'PixelTrackProducer'):
        module.RegionFactoryPSet.RegionPSet.useMultipleScattering = cms.bool(False)

    for module in esproducers_by_type(process,'Chi2ChargeMeasurementEstimatorESProducer'):    
        if not hasattr(module,'MaxDisplacement'):
            module.MaxDisplacement  = cms.double(100.)
        if not hasattr(module,'MaxSagitta'):
            module.MaxSagitta       = cms.double(-1.) 
        if not hasattr(module,'MinimalTolerance'):
            module.MinimalTolerance = cms.double(10.) 
    for module in esproducers_by_type(process,'Chi2MeasurementEstimatorESProducer'):
        if not hasattr(module,'MaxDisplacement'):
            module.MaxDisplacement  = cms.double(100.)
        if not hasattr(module,'MaxSagitta'):
            module.MaxSagitta       = cms.double(-1.) 
        if not hasattr(module,'MinimalTolerance'):
            module.MinimalTolerance = cms.double(10.) 

    for pset in process._Process__psets.values():
        if hasattr(pset,'ComponentType'):
            if (pset.ComponentType == 'CkfBaseTrajectoryFilter'):
                if not hasattr(pset,'minGoodStripCharge'):
                    pset.minGoodStripCharge  = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutNone'))
                if not hasattr(pset,'maxCCCLostHits'):    
                    pset.maxCCCLostHits      = cms.int32(9999)
                if not hasattr(pset,'seedExtension'):
                    pset.seedExtension       = cms.int32(0)
                if not hasattr(pset,'strictSeedExtension'):
                    pset.strictSeedExtension = cms.bool(False)

    from HLTrigger.Configuration.customizeHLTfor2016trackingTemplate import *
    process = customiseFor2016trackingTemplate(process)
    return process
>>>>>>> cms-sw/refs/pull/13062/head

#
# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):
    import os
    cmsswVersion = os.environ['CMSSW_VERSION']

    if cmsswVersion >= "CMSSW_8_0":
#       process = customiseFor12718(process)
        process = customiseFor13062(process)
        pass

    return process
