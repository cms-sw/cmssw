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

    return process
