# Based on the end-of-2023 assumptions for PU65 and 90/fb
# https://indico.cern.ch/event/1237252/contributions/5204534/attachments/2574097/4448011/v4_Prep_for_2023_TSG_Jan17_2023.pdf 
# "mild"(medium) cuts update for HB, while keeping 2022 HE cuts intact 
# To be used in cmsDriver command: 
# --customise RecoParticleFlow/PFClusterProducer/particleFlow_HB2023.customiseHB2023

import FWCore.ParameterSet.Config as cms
def customiseHB2023(process):
    recHB  =[0.4, 0.3, 0.3, 0.3]
    seedHB =[0.6, 0.5, 0.5, 0.5]
    process.particleFlowClusterHBHE.seedFinder.thresholdsByDetector[0].seedingThreshold = seedHB
    process.particleFlowClusterHBHE.initialClusteringStep.thresholdsByDetector[0].gatheringThreshold = recHB
    process.particleFlowClusterHBHE.pfClusterBuilder.recHitEnergyNorms[0].recHitEnergyNorm = recHB
    process.particleFlowClusterHBHE.pfClusterBuilder.positionCalc.logWeightDenominatorByDetector[0].logWeightDenominator = recHB
    process.particleFlowClusterHBHE.pfClusterBuilder.allCellsPositionCalc.logWeightDenominatorByDetector[0].logWeightDenominator = recHB
    process.particleFlowClusterHCAL.pfClusterBuilder.allCellsPositionCalc.logWeightDenominatorByDetector[0].logWeightDenominator = recHB
    process.particleFlowRecHitHBHE.producers[0].qualityTests[0].cuts[0].threshold = recHB

    return(process)
