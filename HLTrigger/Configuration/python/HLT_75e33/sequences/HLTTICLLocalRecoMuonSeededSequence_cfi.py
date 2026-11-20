import FWCore.ParameterSet.Config as cms

from ..modules.hltBarrelLayerClustersEBMuonSeeded_cfi import *
from ..modules.hltBarrelLayerClustersHB_cfi import *
from ..modules.hltHGCalRecHitMuonSeeded_cfi import *
from ..modules.hltHGCalUncalibRecHitMuonSeeded_cfi import *
from ..modules.hltHgcalDigis_cfi import *
from ..modules.hltHgcalDigisMuonSeeded_cfi import *
from ..modules.hltHgcalLayerClustersEEMuonSeeded_cfi import *
from ..modules.hltHgcalLayerClustersHSciMuonSeeded_cfi import *
from ..modules.hltHgcalLayerClustersHSiMuonSeeded_cfi import *
from ..modules.hltMergeLayerClustersMuonSeeded_cfi import *
from ..modules.hltParticleFlowRecHitHBHE_cfi import *
from ..sequences.HLTEcalPFClusteringMuonSeededSequence_cfi import *

HLTTICLLocalRecoMuonSeededSequence = cms.Sequence(
    hltHgcalDigis
    + hltHgcalDigisMuonSeeded
    + hltHGCalUncalibRecHitMuonSeeded
    + hltHGCalRecHitMuonSeeded
    + hltHgcalLayerClustersEEMuonSeeded
    + hltHgcalLayerClustersHSciMuonSeeded
    + hltHgcalLayerClustersHSiMuonSeeded
    + hltMergeLayerClustersMuonSeeded
)

_HLTTICLLocalRecoMuonSeededSequence_withBarrel = cms.Sequence(
    hltHgcalDigis
    + hltHgcalDigisMuonSeeded
    + hltHGCalUncalibRecHitMuonSeeded
    + hltHGCalRecHitMuonSeeded
    + hltHgcalLayerClustersEEMuonSeeded
    + hltHgcalLayerClustersHSciMuonSeeded
    + hltHgcalLayerClustersHSiMuonSeeded
    + HLTEcalPFClusteringMuonSeededSequence
    + hltParticleFlowRecHitHBHE
    + hltBarrelLayerClustersEBMuonSeeded
    + hltBarrelLayerClustersHB
    + hltMergeLayerClustersMuonSeeded
)

from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
ticl_barrel.toReplaceWith(HLTTICLLocalRecoMuonSeededSequence, _HLTTICLLocalRecoMuonSeededSequence_withBarrel)
