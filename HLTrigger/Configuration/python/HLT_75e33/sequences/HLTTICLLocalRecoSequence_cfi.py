import FWCore.ParameterSet.Config as cms

from ..modules.hltMergeLayerClusters_cfi import *
from ..modules.hltHGCalRecHit_cfi import *
from ..modules.hltHGCalUncalibRecHit_cfi import *
from ..sequences.HLTHgcalLayerClustersSequence_cfi import *
# Barrel layer clusters
from ..modules.hltParticleFlowRecHitECALUnseeded_cfi import *
from ..modules.hltParticleFlowRecHitHBHE_cfi import *
from ..modules.hltBarrelLayerClustersEB_cfi import *
from ..modules.hltBarrelLayerClustersHB_cfi import *
from ..sequences.HLTPfRecHitUnseededSequence_cfi import *

from Configuration.ProcessModifiers.alpakaValidationHLT_cff import alpakaValidationHLT
from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel


HLTTICLLocalRecoSequence = cms.Sequence(
        hltHGCalUncalibRecHit+
        hltHGCalRecHit+
        HLTHgcalLayerClustersSequence+
        hltMergeLayerClusters)

#Define a GPU+CPU instance of TICLLocalRecoSequence, to be triggered by 'alpakaValidationHLT' procModifier
_HLTTICLLocalRecoSequence_heterogeneousGPUCPU = cms.Sequence(
        hltHGCalUncalibRecHit+
        hltHGCalRecHit+
        HLTHgcalLayerClustersSequence+
        hltMergeLayerClusters+
        #CPU part: runs dedicated 'SerialSync' modules on CPU
        HLTHgcalLayerClustersSequenceSerialSync+
        hltMergeLayerClustersSerialSync)
alpakaValidationHLT.toReplaceWith(HLTTICLLocalRecoSequence, _HLTTICLLocalRecoSequence_heterogeneousGPUCPU)

_HLTTICLLocalRecoSequence_withBarrel = cms.Sequence(
        hltHGCalUncalibRecHit+
        hltHGCalRecHit+
        HLTHgcalLayerClustersSequence+
        HLTPfRecHitUnseededSequence+
        hltBarrelLayerClustersEB+
        hltBarrelLayerClustersHB+
        hltMergeLayerClusters
)
ticl_barrel.toReplaceWith(HLTTICLLocalRecoSequence, _HLTTICLLocalRecoSequence_withBarrel)
