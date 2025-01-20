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
    # For running HLT offline on Run-3 Data, use "(OnlineBeamSpotESProducer).timeThreshold = 1e6",
    # in order to pick the beamspot that was actually used by the HLT (instead of a "fake" beamspot).
    # These same settings can be used offline for Run-3 Data and Run-3 MC alike.
    # Note: the products of the OnlineBeamSpotESProducer are used only
    #       if the configuration uses "(BeamSpotOnlineProducer).useTransientRecord = True".
    # See CMSHLT-2271 and CMSHLT-2300 for further details.
    for prod in esproducers_by_type(process, 'OnlineBeamSpotESProducer'):
        prod.timeThreshold = int(1e6)

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
def configureFrameSoAESProducers(process):
    """
    Configures the appropriate FrameSoAESProducer based on the pixel topology (Phase1, Phase2, etc.).
    If the corresponding producer is not found, it will add it to the process.
    """
    
    # Define a mapping of pixel topology to corresponding ESProducer and component name
    topology_to_es_producer = {
        'Phase1': ('frameSoAESProducerPhase1', 'FrameSoAPhase1', 'FrameSoAESProducerPhase1@alpaka'),
        'HIonPhase1': ('frameSoAESProducerHIonPhase1', 'FrameSoAPhase1HIonPhase1', 'FrameSoAESProducerHIonPhase1@alpaka'),
        'Phase2': ('frameSoAESProducerPhase2', 'FrameSoAPhase2', 'FrameSoAESProducerPhase2@alpaka'),
        'Phase1Strip': ('frameSoAESProducerPhase1Strip', 'FrameSoAPhase1Strip', 'FrameSoAESProducerPhase1Strip@alpaka'),
    }

    has_alpaka_named_module = any("Alpaka" in name for name in process.__dict__.keys())

    if not has_alpaka_named_module:
        print("No modules with 'Alpaka' in their names found in the process. Skipping configuration of FrameSoAESProducers.")
        return process
    for pixel_topology, (es_name, component_name, producer_type) in topology_to_es_producer.items():
        if not hasattr(process, es_name):
            # If the producer does not exist, create and add it
            #print(f"Adding {es_name} with component name {component_name}")
            setattr(process, es_name, cms.ESProducer(producer_type,
                                                     ComponentName=cms.string(component_name),
                                                     appendToDataLabel=cms.string('')))
        else:
            print(f"{es_name} already configured.")

    return process

def customizeHLTfor47047(process):
    """Migrates many ESProducers to MoveToDeviceCache"""
    import copy
    if hasattr(process, "ecalMultifitParametersSource"):
        del process.ecalMultifitParametersSource
    esProducer = None
    for prod in esproducers_by_type(process, "EcalMultifitParametersHostESProducer@alpaka"):
        if esProducer is not None:
            raise Exception("Assumption of only one EcalMultifitParametersHostESProducer@alpaka in a process broken")
        esProducer = prod
    if esProducer is not None:
        for prod in producers_by_type(process, "EcalUncalibRecHitProducerPortable@alpaka", "alpaka_serial_sync::EcalUncalibRecHitProducerPortable"):
            for attr in ["EBtimeFitParameters", "EEtimeFitParameters", "EBamplitudeFitParameters", "EEamplitudeFitParameters"]:
                setattr(prod, attr, copy.deepcopy(getattr(esProducer, attr)))
        delattr(process, esProducer.label())

    for prod in producers_by_type(process, "HBHERecHitProducerPortable@alpaka", "alpaka_serial_sync::HBHERecHitProducerPortable"):
        pulseOffsetLabel = prod.mahiPulseOffSets.getModuleLabel()
        if hasattr(process, pulseOffsetLabel):
            esProducer = getattr(process, pulseOffsetLabel)
            prod.pulseOffsets = copy.deepcopy(esProducer.pulseOffsets)
        del prod.mahiPulseOffSets
    for prod in list(esproducers_by_type(process, "HcalMahiPulseOffsetsESProducer@alpaka")):
        delattr(process, prod.label())

    for prod in producers_by_type(process, "PFClusterSoAProducer@alpaka", "alpaka_serial_sync::PFClusterSoAProducer"):
        clusterParamsLabel = prod.pfClusterParams.getModuleLabel()
        if hasattr(process, clusterParamsLabel):
            esProducer = getattr(process, clusterParamsLabel)
            for attr in ["seedFinder", "initialClusteringStep", "pfClusterBuilder"]:
                setattr(prod, attr, copy.deepcopy(getattr(esProducer, attr).copy()))
        del prod.pfClusterParams
    for prod in list(esproducers_by_type(process, "PFClusterParamsESProducer@alpaka")):
        delattr(process, prod.label())

    if hasattr(process, "hltESSJobConfigurationGPURecord"):
        del process.hltESSJobConfigurationGPURecord

    return process
        
def customizeHLTfor47107(process):
    """Remove unneeded parameters from the HLT menu"""

    for prod in producers_by_type(process, 'TrackProducer'):
        if hasattr(prod, 'alias'):
            delattr(prod, 'alias')

    for prod in producers_by_type(process, 'GsfTrackProducer'):
        if hasattr(prod, 'producer'):
            delattr(prod, 'producer')

    return process

# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):

    process = customiseForOffline(process)
    process = configureFrameSoAESProducers(process)
     
    # add call to action function in proper order: newest last!
    # process = customiseFor12718(process)

    process = customizeHLTfor46935(process)
    process = customizeHLTfor47017(process)
    process = customizeHLTfor47079(process)
    process = customizeHLTfor47047(process)
    process = customizeHLTfor47107(process)

    return process
