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

def customiseFor21810(process):
    for producer in producers_by_type(process, "CaloTowersCreator"):
        producer.HcalPhase = cms.int32(0)
        producer.HcalCollapsed = cms.bool(True)
        producer.HESThreshold1 = cms.double(0.8)
        producer.HESThreshold  = cms.double(0.8)
        producer.HEDThreshold1 = cms.double(0.8)
        producer.HEDThreshold  = cms.double(0.8)
    return process

def customiseFor21845(process):

    for producer in producers_by_type(process, "PFClusterProducer"):
        if producer.seedFinder.thresholdsByDetector[0].detector.value()=='HCAL_BARREL1':
            producer.seedFinder.thresholdsByDetector[0].depths = cms.vint32(1, 2, 3, 4)
            producer.seedFinder.thresholdsByDetector[0].seedingThreshold = cms.vdouble(1.0, 1.0, 1.0, 1.0)
            producer.seedFinder.thresholdsByDetector[0].seedingThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0)

        if producer.seedFinder.thresholdsByDetector[1].detector.value()=='HCAL_ENDCAP':
            producer.seedFinder.thresholdsByDetector[1].depths = cms.vint32(1, 2, 3, 4, 5, 6, 7)
            producer.seedFinder.thresholdsByDetector[1].seedingThreshold = cms.vdouble(1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1)
            producer.seedFinder.thresholdsByDetector[1].seedingThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

            #############

        if producer.initialClusteringStep.thresholdsByDetector[0].detector.value()=='HCAL_BARREL1':
            producer.initialClusteringStep.thresholdsByDetector[0].depths = cms.vint32(1, 2, 3, 4)
            producer.initialClusteringStep.thresholdsByDetector[0].gatheringThreshold = cms.vdouble(0.8, 0.8, 0.8, 0.8)
            producer.initialClusteringStep.thresholdsByDetector[0].gatheringThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0)

        if producer.initialClusteringStep.thresholdsByDetector[1].detector.value()=='HCAL_ENDCAP':
            producer.initialClusteringStep.thresholdsByDetector[1].depths = cms.vint32(1, 2, 3, 4, 5, 6, 7)
            producer.initialClusteringStep.thresholdsByDetector[1].gatheringThreshold = cms.vdouble(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
            producer.initialClusteringStep.thresholdsByDetector[1].gatheringThresholdPt = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

            #############

        if producer.pfClusterBuilder.recHitEnergyNorms[0].detector.value()=='HCAL_BARREL1':
            producer.pfClusterBuilder.recHitEnergyNorms[0].depths = cms.vint32(1, 2, 3, 4)
            producer.pfClusterBuilder.recHitEnergyNorms[0].recHitEnergyNorm = cms.vdouble(0.8, 0.8, 0.8, 0.8)

        if producer.pfClusterBuilder.recHitEnergyNorms[1].detector.value()=='HCAL_ENDCAP':
            producer.pfClusterBuilder.recHitEnergyNorms[1].depths = cms.vint32(1, 2, 3, 4, 5, 6, 7)
            producer.pfClusterBuilder.recHitEnergyNorms[1].recHitEnergyNorm = cms.vdouble(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8)

            #############

    for producer in producers_by_type(process, "PFRecHitProducer"):
        if producer.producers[0].name.value()=='PFHBHERecHitCreator':
            producer.producers[0].qualityTests[0].cuts = cms.VPSet(
                cms.PSet(),
                cms.PSet()
                )
            producer.producers[0].qualityTests[0].cuts[0].depth = cms.vint32(1, 2, 3, 4)
            producer.producers[0].qualityTests[0].cuts[0].threshold = cms.vdouble(0.8, 0.8, 0.8, 0.8)
            producer.producers[0].qualityTests[0].cuts[0].detectorEnum = cms.int32(1)

            producer.producers[0].qualityTests[0].cuts[1].depth = cms.vint32(1, 2, 3, 4, 5, 6, 7)
            producer.producers[0].qualityTests[0].cuts[1].threshold = cms.vdouble(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
            producer.producers[0].qualityTests[0].cuts[1].detectorEnum = cms.int32(2)

    for producer in producers_by_type(process, "PFRecHitProducer"):
        if producer.producers[0].name.value()=='PFHFRecHitCreator':
            producer.producers[0].qualityTests[1].cuts = cms.VPSet(
                cms.PSet()
                )
            producer.producers[0].qualityTests[1].cuts[0].depth = cms.vint32(1,2)
            producer.producers[0].qualityTests[1].cuts[0].threshold = cms.vdouble(1.2,1.8)
            producer.producers[0].qualityTests[1].cuts[0].detectorEnum = cms.int32(4)

    return process


def customiseFor21664_forMahiOn(process):
    for producer in producers_by_type(process, "HBHEPhase1Reconstructor"):
        producer.algorithm.useMahi   = cms.bool(True)
        producer.algorithm.useM2     = cms.bool(False)
        producer.algorithm.useM3     = cms.bool(False)
    return process

def customiseFor21664_forMahiOnM2only(process):
    for producer in producers_by_type(process, "HBHEPhase1Reconstructor"):
      if (producer.algorithm.useM2 == cms.bool(True)):
        producer.algorithm.useMahi   = cms.bool(True)
        producer.algorithm.useM2     = cms.bool(False)
        producer.algorithm.useM3     = cms.bool(False)
    return process

# Needs the ESProducer for HcalTimeSlewRecord
def customiseFor21733(process):
    process.load('CalibCalorimetry.HcalPlugins.HcalTimeSlew_cff')
    return process

# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):

    # add call to action function in proper order: newest last!
    # process = customiseFor12718(process)
        
    process = customiseFor21733(process)

    process = customiseFor21810(process)

    process = customiseFor21845(process)

    return process
