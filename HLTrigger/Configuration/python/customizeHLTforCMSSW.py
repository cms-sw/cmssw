import FWCore.ParameterSet.Config as cms

# modifiers
from Configuration.ProcessModifiers.gpu_cff import gpu

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

def customiseHCALFor2018Input(process):
    """Customise the HLT to run on Run 2 data/MC using the old readout for the HCAL barel"""

    for producer in producers_by_type(process, "HBHEPhase1Reconstructor"):
        # switch on the QI8 processing for 2018 HCAL barrel
        producer.processQIE8 = True

    # adapt CaloTowers threshold for 2018 HCAL barrel with only one depth
    for producer in producers_by_type(process, "CaloTowersCreator"):
        producer.HBThreshold1  = 0.7
        producer.HBThreshold2  = 0.7
        producer.HBThreshold   = 0.7

    # adapt Particle Flow threshold for 2018 HCAL barrel with only one depth
    from RecoParticleFlow.PFClusterProducer.particleFlowClusterHBHE_cfi import _thresholdsHB, _thresholdsHEphase1, _seedingThresholdsHB

    logWeightDenominatorHCAL2018 = cms.VPSet(
        cms.PSet(
            depths = cms.vint32(1, 2, 3, 4),
            detector = cms.string('HCAL_BARREL1'),
            logWeightDenominator = _thresholdsHB
        ),
        cms.PSet(
            depths = cms.vint32(1, 2, 3, 4, 5, 6, 7),
            detector = cms.string('HCAL_ENDCAP'),
            logWeightDenominator = _thresholdsHEphase1
        )
    )

    for producer in producers_by_type(process, "PFRecHitProducer"):
        if producer.producers[0].name.value() == 'PFHBHERecHitCreator':
            producer.producers[0].qualityTests[0].cuts[0].threshold = _thresholdsHB

    for producer in producers_by_type(process, "PFClusterProducer"):
        if producer.seedFinder.thresholdsByDetector[0].detector.value() == 'HCAL_BARREL1':
            producer.seedFinder.thresholdsByDetector[0].seedingThreshold = _seedingThresholdsHB
            producer.initialClusteringStep.thresholdsByDetector[0].gatheringThreshold = _thresholdsHB
            producer.pfClusterBuilder.recHitEnergyNorms[0].recHitEnergyNorm = _thresholdsHB
            producer.pfClusterBuilder.positionCalc.logWeightDenominatorByDetector = logWeightDenominatorHCAL2018
            producer.pfClusterBuilder.allCellsPositionCalc.logWeightDenominatorByDetector = logWeightDenominatorHCAL2018

    for producer in producers_by_type(process, "PFMultiDepthClusterProducer"):
        producer.pfClusterBuilder.allCellsPositionCalc.logWeightDenominatorByDetector = logWeightDenominatorHCAL2018

    # done
    return process

def customiseFor2017DtUnpacking(process):
    """Adapt the HLT to run the legacy DT unpacking
    for pre2018 data/MC workflows as the default"""

    if hasattr(process,'hltMuonDTDigis'):
        process.hltMuonDTDigis = cms.EDProducer( "DTUnpackingModule",
            useStandardFEDid = cms.bool( True ),
            maxFEDid = cms.untracked.int32( 779 ),
            inputLabel = cms.InputTag( "rawDataCollector" ),
            minFEDid = cms.untracked.int32( 770 ),
            dataType = cms.string( "DDU" ),
            readOutParameters = cms.PSet(
                localDAQ = cms.untracked.bool( False ),
                debug = cms.untracked.bool( False ),
                rosParameters = cms.PSet(
                    localDAQ = cms.untracked.bool( False ),
                    debug = cms.untracked.bool( False ),
                    writeSC = cms.untracked.bool( True ),
                    readDDUIDfromDDU = cms.untracked.bool( True ),
                    readingDDU = cms.untracked.bool( True ),
                    performDataIntegrityMonitor = cms.untracked.bool( False )
                    ),
                performDataIntegrityMonitor = cms.untracked.bool( False )
                ),
            dqmOnly = cms.bool( False )
        )

    return process

def customisePixelGainForRun2Input(process):
    """Customise the HLT to run on Run 2 data/MC using the old definition of the pixel calibrations

    Up to 11.0.x, the pixel calibarations were fully specified in the configuration:
        VCaltoElectronGain      =   47
        VCaltoElectronGain_L1   =   50
        VCaltoElectronOffset    =  -60
        VCaltoElectronOffset_L1 = -670

    Starting with 11.1.x, the calibrations for Run 3 were moved to the conditions, leaving in the configuration only:
        VCaltoElectronGain      =    1
        VCaltoElectronGain_L1   =    1
        VCaltoElectronOffset    =    0
        VCaltoElectronOffset_L1 =    0

    Since the conditions for Run 2 have not been updated to the new scheme, the HLT configuration needs to be reverted.
    """
    # revert the Pixel parameters to be compatible with the Run 2 conditions
    for producer in producers_by_type(process, "SiPixelClusterProducer"):
        producer.VCaltoElectronGain      =   47
        producer.VCaltoElectronGain_L1   =   50
        producer.VCaltoElectronOffset    =  -60
        producer.VCaltoElectronOffset_L1 = -670

    for producer in producers_by_type(process, "SiPixelRawToClusterCUDA"):
        producer.isRun2 = True

    return process

def customisePixelL1ClusterThresholdForRun2Input(process):
    # revert the pixel Layer 1 cluster threshold to be compatible with Run2:
    for producer in producers_by_type(process, "SiPixelClusterProducer"):
        if hasattr(producer,"ClusterThreshold_L1"):
            producer.ClusterThreshold_L1 = 2000
    for producer in producers_by_type(process, "SiPixelRawToClusterCUDA"):
        if hasattr(producer,"clusterThreshold_layer1"):
            producer.clusterThreshold_layer1 = 2000

    return process

def customiseCTPPSFor2018Input(process):
    for prod in producers_by_type(process, 'CTPPSGeometryESModule'):
        prod.isRun2 = True
    for prod in producers_by_type(process, 'CTPPSPixelRawToDigi'):
        prod.isRun3 = False

    return process

def customiseEGammaRecoFor2018Input(process):
    for prod in producers_by_type(process, 'PFECALSuperClusterProducer'):
        if hasattr(prod, 'regressionConfig'):
            prod.regressionConfig.regTrainedWithPS = cms.bool(False)

    return process

def customiseFor2018Input(process):
    """Customise the HLT to run on Run 2 data/MC"""
    process = customisePixelGainForRun2Input(process)
    process = customisePixelL1ClusterThresholdForRun2Input(process)
    process = customiseHCALFor2018Input(process)
    process = customiseCTPPSFor2018Input(process)
    process = customiseEGammaRecoFor2018Input(process)

    return process


def customiseFor37231(process):
    """ Customisation to fix the typo of Reccord in PR 37231 (https://github.com/cms-sw/cmssw/pull/37231) """

    for prod in producers_by_type(process, 'DeDxEstimatorProducer'):
        if hasattr(prod, 'Reccord'):
            prod.Record = prod.Reccord
            delattr(prod, 'Reccord')

    return process

def customiseFor37756(process):
    """https://github.com/cms-sw/cmssw/pull/37756
    Removal of use_preshower parameter from PFECALSuperClusterProducer
    """
    for prod in producers_by_type(process, 'PFECALSuperClusterProducer'):
        if hasattr(prod, 'use_preshower'):
            delattr(prod, 'use_preshower')

    return process

def customiseFor37646(process):
    """ Customisation to remove a renamed parameter in HLTScoutingPFProducer
     from PR 37646 (https://github.com/cms-sw/cmssw/pull/37646)
    """
    for prod in producers_by_type(process, 'HLTScoutingPFProducer'):
        if hasattr(prod, 'doTrackRelVars'):
            delattr(prod, 'doTrackRelVars')
            
    return process

  
def customiseForOffline(process):
#   https://its.cern.ch/jira/browse/CMSHLT-2271
    for prod in producers_by_type(process, 'BeamSpotOnlineProducer'):
        prod.useTransientRecord = False

    return process


# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):

    # if the gpu modifier is enabled, make the Pixel, ECAL and HCAL reconstruction offloadable to a GPU
    from HLTrigger.Configuration.customizeHLTforPatatrack import customizeHLTforPatatrack
    gpu.makeProcessModifier(customizeHLTforPatatrack).apply(process)
    process = customiseForOffline(process)

    # add call to action function in proper order: newest last!
    # process = customiseFor12718(process)

    process = customiseFor37231(process)
    process = customiseFor37646(process)
    process = customiseFor37756(process)
    
    return process
