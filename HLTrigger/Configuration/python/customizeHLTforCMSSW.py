import FWCore.ParameterSet.Config as cms

# helper functions
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



def customiseForOffline(process):
    # For running HLT offline and relieve the strain on Frontier so it will no longer inject a
    # transaction id which tells Frontier to add a unique "&freshkey" to many query URLs.
    # That was intended as a feature to only be used by the Online HLT, to guarantee that fresh conditions
    # from the database were loaded at each Lumi section
    # Seee CMSHLT-3123 for further details
    if hasattr(process, 'GlobalTag'):
        # Set ReconnectEachRun and RefreshEachRun to False
        process.GlobalTag.ReconnectEachRun = cms.untracked.bool(False)
        process.GlobalTag.RefreshEachRun = cms.untracked.bool(False)

        if hasattr(process.GlobalTag, 'toGet'):
            # Filter out PSet objects containing only 'record' and 'refreshTime'
            process.GlobalTag.toGet = [
                pset for pset in process.GlobalTag.toGet
                if set(pset.parameterNames_()) != {'record', 'refreshTime'}
            ]

    return process


def customizeHLTfor46935(process):
    """Changes parameter names of EcalUncalibRecHitSoAToLegacy producer"""
    for prod in producers_by_type(process, 'EcalUncalibRecHitSoAToLegacy'):
        if hasattr(prod, 'uncalibRecHitsPortableEB'):
            prod.inputCollectionEB = prod.uncalibRecHitsPortableEB
            delattr(prod, 'uncalibRecHitsPortableEB')
        if hasattr(prod, 'uncalibRecHitsPortableEE'):
            prod.inputCollectionEE = prod.uncalibRecHitsPortableEE
            delattr(prod, 'uncalibRecHitsPortableEE')
        if hasattr(prod, 'recHitsLabelCPUEB'):
            prod.outputLabelEB = prod.recHitsLabelCPUEB
            delattr(prod, 'recHitsLabelCPUEB')
        if hasattr(prod, 'recHitsLabelCPUEE'):
            prod.outputLabelEE = prod.recHitsLabelCPUEE
            delattr(prod, 'recHitsLabelCPUEE')
    return process

def customiseHLTFor46647(process):
    for prod in producers_by_type(process, 'CtfSpecialSeedGenerator'):
        if hasattr(prod, "DontCountDetsAboveNClusters"):
            value = prod.DontCountDetsAboveNClusters.value()
            delattr(prod, "DontCountDetsAboveNClusters")
            # Replace it with cms.uint32
            prod.DontCountDetsAboveNClusters = cms.uint32(value)


def customizeHLTfor47017(process):
    """Remove unneeded parameters from the HLT menu"""
    for prod in producers_by_type(process, 'MaskedMeasurementTrackerEventProducer'):
        if hasattr(prod, 'OnDemand'):
            delattr(prod, 'OnDemand')

    for prod in producers_by_type(process, 'HcalHaloDataProducer'):
        if hasattr(prod, 'HcalMaxMatchingRadiusParam'):
            delattr(prod, 'HcalMaxMatchingRadiusParam')
        if hasattr(prod, 'HcalMinMatchingRadiusParam'):
            delattr(prod, 'HcalMinMatchingRadiusParam')

    for prod in producers_by_type(process, 'SiPixelRecHitConverter'):
        if hasattr(prod, 'VerboseLevel'):
            delattr(prod, 'VerboseLevel')

def customizeHLTforXXX(process):
    if not hasattr(process, 'HLTRecoPixelTracksSequence'):
        return process

    process.frameSoAESProducerPhase1 = cms.ESProducer('FrameSoAESProducerPhase1@alpaka',
      ComponentName = cms.string('FrameSoAPhase1'),
      appendToDataLabel = cms.string(''),
      alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
      )
    )

    for producer in producers_by_type(process, "CAHitNtupletAlpakaPhase1@alpaka"):
        #print("entered the producers loop")
        if hasattr(producer, "CPE"):
            print("found CPE stuff")
            delattr(producer, "CPE")
        if not hasattr(producer, 'frameSoA'):
            setattr(producer, 'frameSoA', cms.string('FrameSoAPhase1'))

    for producer in producers_by_type(process, "alpaka_serial_sync::CAHitNtupletAlpakaPhase1"):
        #print("entered the producers loop")
        if hasattr(producer, "CPE"):
            print("found CPE stuff")
            delattr(producer, "CPE")
        if not hasattr(producer, 'frameSoA'):
            setattr(producer, 'frameSoA', cms.string('FrameSoAPhase1'))
    
    return process


def customizeHLTfor47079(process):
    """Remove unneeded parameters from the HLT menu"""
    for filt in filters_by_type(process, 'PrimaryVertexObjectFilter'):
        if hasattr(filt, 'filterParams') and hasattr(filt.filterParams, 'pvSrc'):
            del filt.filterParams.pvSrc  # Remove the pvSrc parameter

    for prod in producers_by_type(process, 'HcalHitReconstructor'):
        # Remove useless parameters
        if hasattr(prod,'setHSCPFlags'):
            delattr(prod,'setHSCPFlags')

        if hasattr(prod,'setPulseShapeFlags'):
            delattr(prod,'setPulseShapeFlags')
                    
    return process

def customizeHLTfor47047(process):
    """Migrates many ESProducers to MoveToDeviceCache"""
>>>>>>> 82c6e3e3d74 (Moving histo out of hits and new FrameSoA)
    import copy
    for esprod in list(esproducers_by_type(process, "OnlineBeamSpotESProducer")):
        delattr(process, esprod.label())

    for edprod in producers_by_type(process, "BeamSpotOnlineProducer"):
        if hasattr(edprod, 'useTransientRecord'):
            setattr(edprod, 'useBSOnlineRecords', copy.deepcopy(getattr(edprod, 'useTransientRecord')))
            delattr(edprod, 'useTransientRecord')
    
    return process

# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):

    process = customiseForOffline(process)

    # add call to action function in proper order: newest last!
    # process = customiseFor12718(process)

    process = customizeHLTfor47378(process)
    process = customizeHLTforXXX(process)
   
    return process
