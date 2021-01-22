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

from RecoParticleFlow.PFClusterProducer.particleFlowClusterHCAL_cfi import _thresholdsHB
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHBHE_cfi import _seedingThresholdsHB, _thresholdsHB
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHBHE_cfi import _thresholdsHB as _thresholdsHBRec

from RecoParticleFlow.PFClusterProducer.particleFlowClusterHCAL_cfi import _thresholdsHEphase1 as _thresholdsHEphase1HCAL
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHBHE_cfi import _seedingThresholdsHEphase1, _thresholdsHEphase1
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHBHE_cfi import _thresholdsHEphase1 as _thresholdsHEphase1Rec


logWeightDenominatorHCAL2018 = cms.VPSet(
    cms.PSet(
        depths = cms.vint32(1, 2, 3, 4),
        detector = cms.string('HCAL_BARREL1'),
        logWeightDenominator = _thresholdsHB
        ),
    cms.PSet(
        depths = cms.vint32(
    1, 2, 3, 4, 5, 6, 7
        ),
        detector = cms.string('HCAL_ENDCAP'),
        logWeightDenominator = _thresholdsHEphase1HCAL
    )
)


def synchronizeHCALHLTofflineRun3on2018data(process):
    # this function bring back the Run3 menu to a Run2-2018 like meny, for testing in data 2018

    #----------------------------------------------------------------------------------------------------------
    # adapt threshold for HB  - in 2018 only one depth

    for producer in producers_by_type(process, "PFClusterProducer"):
        if producer.seedFinder.thresholdsByDetector[0].detector.value() == 'HCAL_BARREL1':
            producer.seedFinder.thresholdsByDetector[0].seedingThreshold              = _seedingThresholdsHB
            producer.initialClusteringStep.thresholdsByDetector[0].gatheringThreshold = _thresholdsHB
            producer.pfClusterBuilder.recHitEnergyNorms[0].recHitEnergyNorm           = _thresholdsHB

            producer.pfClusterBuilder.positionCalc.logWeightDenominatorByDetector = logWeightDenominatorHCAL2018
            producer.pfClusterBuilder.allCellsPositionCalc.logWeightDenominatorByDetector = logWeightDenominatorHCAL2018

    for producer in producers_by_type(process, "PFMultiDepthClusterProducer"):
        producer.pfClusterBuilder.allCellsPositionCalc.logWeightDenominatorByDetector = logWeightDenominatorHCAL2018

    for producer in producers_by_type(process, "PFRecHitProducer"):
        if producer.producers[0].name.value() == 'PFHBHERecHitCreator':
            producer.producers[0].qualityTests[0].cuts[0].threshold = _thresholdsHBRec

    for producer in producers_by_type(process, "CaloTowersCreator"):
        producer.HBThreshold1  = cms.double(0.7)
        producer.HBThreshold2  = cms.double(0.7)
        producer.HBThreshold   = cms.double(0.7)

    #--------------------------------------------------------
    # switch on the QI8 processing as in HB-Run2 (in Run3 we have only QIE11)
    for producer in producers_by_type(process, "HBHEPhase1Reconstructor"):
        producer.processQIE8 = cms.bool( True )
        producer.setNoiseFlagsQIE8 = cms.bool( True )
        producer.setPulseShapeFlagsQIE8 = cms.bool( True )

    #----------------------------------------------------------
    # Use 1+8p fit (PR29617) and apply HB- correction (PR26177)
    for producer in producers_by_type(process, "HBHEPhase1Reconstructor"):
        producer.algorithm.applyLegacyHBMCorrection = cms.bool( True )
        producer.algorithm.chiSqSwitch = cms.double(15.0)

    return process

def synchronizeHCALHLTofflineRun2(process):
    # this function bring forward the sw changes of Run3 to 2018 data starting from a Run2-2018 like menu

    #-----------------------------------------------------------------------------------------------------------
    # A) remove collapser from sequence
    process.hltHbhereco = process.hltHbhePhase1Reco.clone()
    process.HLTDoLocalHcalSequence      = cms.Sequence( process.hltHcalDigis + process.hltHbhereco + process.hltHfprereco + process.hltHfreco + process.hltHoreco )
    process.HLTStoppedHSCPLocalHcalReco = cms.Sequence( process.hltHcalDigis + process.hltHbhereco )
    process.HLTDoLocalHcalWithTowerSequence = cms.Sequence( process.hltHcalDigis + process.hltHbhereco + process.hltHfprereco + process.hltHfreco + process.hltHoreco + process.hltTowerMakerForAll )


    #----------------------------------------------------------------------------------------------------------
    # B) adapt threshold following removal of the collapser
    # note this is done only for HE

    for producer in producers_by_type(process, "PFClusterProducer"):
        if producer.seedFinder.thresholdsByDetector[1].detector.value() == 'HCAL_ENDCAP':
            producer.seedFinder.thresholdsByDetector[1].seedingThreshold              = _seedingThresholdsHEphase1
            producer.initialClusteringStep.thresholdsByDetector[1].gatheringThreshold = _thresholdsHEphase1
            producer.pfClusterBuilder.recHitEnergyNorms[1].recHitEnergyNorm           = _thresholdsHEphase1

            del producer.pfClusterBuilder.positionCalc.logWeightDenominator
            producer.pfClusterBuilder.positionCalc.logWeightDenominatorByDetector = logWeightDenominatorHCAL2018
            del producer.pfClusterBuilder.allCellsPositionCalc.logWeightDenominator
            producer.pfClusterBuilder.allCellsPositionCalc.logWeightDenominatorByDetector = logWeightDenominatorHCAL2018

    for producer in producers_by_type(process, "PFMultiDepthClusterProducer"):
        del producer.pfClusterBuilder.allCellsPositionCalc.logWeightDenominator
        producer.pfClusterBuilder.allCellsPositionCalc.logWeightDenominatorByDetector = logWeightDenominatorHCAL2018

    for producer in producers_by_type(process, "PFRecHitProducer"):
        if producer.producers[0].name.value() == 'PFHBHERecHitCreator':
            producer.producers[0].qualityTests[0].cuts[1].threshold = _thresholdsHEphase1Rec

    for producer in producers_by_type(process, "CaloTowersCreator"):
        producer.HcalPhase     = cms.int32(1)
        producer.HESThreshold1 = cms.double(0.1)
        producer.HESThreshold  = cms.double(0.2)
        producer.HEDThreshold1 = cms.double(0.1)
        producer.HEDThreshold  = cms.double(0.2)

    #--------------------------------------------------------
    # C) add arrival time following PR 26270 (emulate what we will do in Run3 at HLT)
    # (unused HLT quantity, set to false to save CPU)
    for producer in producers_by_type(process, "HBHEPhase1Reconstructor"):
        producer.algorithm.calculateArrivalTime  = cms.bool(False)

    #--------------------------------------------------------
    # D) 3->8 pulse fit for PR 25469 (emulate what we will do in Run3 at HLT)
    for producer in producers_by_type(process, "HBHEPhase1Reconstructor"):
        producer.use8ts = cms.bool(True)
        producer.algorithm.dynamicPed = cms.bool(False)
        producer.algorithm.activeBXs = cms.vint32(-3, -2, -1, 0, 1, 2, 3, 4)

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

    return process


def customiseFor2018Input(process):
    """Customise the HLT to run on Run 2 data/MC"""
    process = customisePixelGainForRun2Input(process)
    process = synchronizeHCALHLTofflineRun3on2018data(process)

    return process


# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):

    # add call to action function in proper order: newest last!
    # process = customiseFor12718(process)

    return process
