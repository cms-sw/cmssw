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


from RecoParticleFlow.PFClusterProducer.particleFlowClusterHBHE_cfi import _seedingThresholdsHEphase1, _thresholdsHEphase1
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHCAL_cfi import _thresholdsHEphase1 as _thresholdsHEphase1HCAL
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHBHE_cfi import _thresholdsHEphase1 as _thresholdsHEphase1Rec

def customiseForUncollapsed(process):
    for producer in producers_by_type(process, "PFClusterProducer"):
        if producer.seedFinder.thresholdsByDetector[1].detector.value() == 'HCAL_ENDCAP':
            producer.seedFinder.thresholdsByDetector[1].seedingThreshold              = _seedingThresholdsHEphase1
            producer.initialClusteringStep.thresholdsByDetector[1].gatheringThreshold = _thresholdsHEphase1
            producer.pfClusterBuilder.recHitEnergyNorms[1].recHitEnergyNorm           = _thresholdsHEphase1
            producer.pfClusterBuilder.positionCalc.logWeightDenominatorByDetector[1].logWeightDenominator = _thresholdsHEphase1
            producer.pfClusterBuilder.allCellsPositionCalc.logWeightDenominatorByDetector[1].logWeightDenominator = _thresholdsHEphase1

    for producer in producers_by_type(process, "PFMultiDepthClusterProducer"):
        producer.pfClusterBuilder.allCellsPositionCalc.logWeightDenominatorByDetector[1].logWeightDenominator = _thresholdsHEphase1HCAL
    
    for producer in producers_by_type(process, "PFRecHitProducer"):
        if producer.producers[0].name.value() == 'PFHBHERecHitCreator':
            producer.producers[0].qualityTests[0].cuts[1].threshold = _thresholdsHEphase1Rec
    
    for producer in producers_by_type(process, "CaloTowersCreator"):
        producer.HcalPhase     = cms.int32(1)
        producer.HESThreshold1 = cms.double(0.1)
        producer.HESThreshold  = cms.double(0.2)
        producer.HEDThreshold1 = cms.double(0.1)
        producer.HEDThreshold  = cms.double(0.2)


    #remove collapser from sequence
    process.hltHbhereco = process.hltHbhePhase1Reco.clone()
    process.HLTDoLocalHcalSequence      = cms.Sequence( process.hltHcalDigis + process.hltHbhereco + process.hltHfprereco + process.hltHfreco + process.hltHoreco )
    process.HLTStoppedHSCPLocalHcalReco = cms.Sequence( process.hltHcalDigis + process.hltHbhereco )


    return process    


def customiseFor21664_forMahiOn(process):
    for producer in producers_by_type(process, "HBHEPhase1Reconstructor"):
        producer.algorithm.useMahi   = cms.bool(True)
        producer.algorithm.useM2     = cms.bool(False)
        producer.algorithm.useM3     = cms.bool(False)
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



# particleFlowRechitECAL new default value "false" flag to be added
def customiseForEcalTestPR22254Default(process):

    for hltParticleFlowRecHitECAL in ['hltParticleFlowRecHitECALUnseeded', 'hltParticleFlowRecHitECALL1Seeded', 'hltParticleFlowRecHitECALForMuonsMF', 'hltParticleFlowRecHitECALForTkMuonsMF']: 
        if hasattr(process,hltParticleFlowRecHitECAL):                                                 
            module = getattr(process,hltParticleFlowRecHitECAL)

            for producer in module.producers: 
                if hasattr(producer,'srFlags'):
                    producer.srFlags = cms.InputTag("")
                if hasattr(producer,'qualityTests'):
                    for qualityTest in producer.qualityTests:
                        if hasattr(qualityTest,'thresholds'):
                            qualityTest.applySelectionsToAllCrystals = cms.bool(True)
                        
    return process



# 
# The three different set of thresholds will be used to study
# possible new thresholds of pfrechits and effects on high level objects
# The values proposed (A, B, C) are driven by expected noise levels
#

# Test thresholds for particleFlowRechitECAL   ~ 0.5 sigma
def customiseForEcalTestPR22254thresholdA(process):
    from Configuration.Eras.Modifier_run2_ECAL_2017_cff import run2_ECAL_2017
    from RecoParticleFlow.PFClusterProducer.particleFlowZeroSuppressionECAL_cff import _particle_flow_zero_suppression_ECAL_2018_A

    for hltParticleFlowRecHitECAL in ['hltParticleFlowRecHitECALUnseeded', 'hltParticleFlowRecHitECALL1Seeded', 'hltParticleFlowRecHitECALForMuonsMF', 'hltParticleFlowRecHitECALForTkMuonsMF']: 
        if hasattr(process,hltParticleFlowRecHitECAL):                                                 
            module = getattr(process,hltParticleFlowRecHitECAL)

            for producer in module.producers: 
                if hasattr(producer,'srFlags'):
                    producer.srFlags = cms.InputTag("")
                if hasattr(producer,'qualityTests'):
                    for qualityTest in producer.qualityTests:
                        if hasattr(qualityTest,'thresholds'):
                            qualityTest.thresholds = _particle_flow_zero_suppression_ECAL_2018_A.thresholds 
                            qualityTest.applySelectionsToAllCrystals = cms.bool(True)
                        
    return process

                   



# Test thresholds for particleFlowRechitECAL   ~ 1 sigma
def customiseForEcalTestPR22254thresholdB(process):
    from Configuration.Eras.Modifier_run2_ECAL_2017_cff import run2_ECAL_2017
    from RecoParticleFlow.PFClusterProducer.particleFlowZeroSuppressionECAL_cff import _particle_flow_zero_suppression_ECAL_2018_B

    for hltParticleFlowRecHitECAL in ['hltParticleFlowRecHitECALUnseeded', 'hltParticleFlowRecHitECALL1Seeded', 'hltParticleFlowRecHitECALForMuonsMF', 'hltParticleFlowRecHitECALForTkMuonsMF']: 
        if hasattr(process,hltParticleFlowRecHitECAL):                                                 
            module = getattr(process,hltParticleFlowRecHitECAL)

            for producer in module.producers: 
                if hasattr(producer,'srFlags'):
                    producer.srFlags = cms.InputTag("")
                if hasattr(producer,'qualityTests'):
                    for qualityTest in producer.qualityTests:
                        if hasattr(qualityTest,'thresholds'):
                            qualityTest.thresholds = _particle_flow_zero_suppression_ECAL_2018_B.thresholds 
                            qualityTest.applySelectionsToAllCrystals = cms.bool(True)
                        
    return process




# Test thresholds for particleFlowRechitECAL   ~ 2 sigma
def customiseForEcalTestPR22254thresholdC(process):
    from Configuration.Eras.Modifier_run2_ECAL_2017_cff import run2_ECAL_2017
    from RecoParticleFlow.PFClusterProducer.particleFlowZeroSuppressionECAL_cff import _particle_flow_zero_suppression_ECAL_2018_C

    for hltParticleFlowRecHitECAL in ['hltParticleFlowRecHitECALUnseeded', 'hltParticleFlowRecHitECALL1Seeded', 'hltParticleFlowRecHitECALForMuonsMF', 'hltParticleFlowRecHitECALForTkMuonsMF']: 
        if hasattr(process,hltParticleFlowRecHitECAL):                                                 
            module = getattr(process,hltParticleFlowRecHitECAL)

            for producer in module.producers: 
                if hasattr(producer,'srFlags'):
                    producer.srFlags = cms.InputTag("")
                if hasattr(producer,'qualityTests'):
                    for qualityTest in producer.qualityTests:
                        if hasattr(qualityTest,'thresholds'):
                            qualityTest.thresholds = _particle_flow_zero_suppression_ECAL_2018_C.thresholds 
                            qualityTest.applySelectionsToAllCrystals = cms.bool(True)
                        
    return process





# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):

    # add call to action function in proper order: newest last!
    # process = customiseFor12718(process)

    return process
