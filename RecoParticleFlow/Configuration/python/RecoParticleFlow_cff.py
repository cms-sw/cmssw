import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.towerMakerPF_cff import *
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import *
from RecoParticleFlow.PFTracking.particleFlowTrack_cff import *
from RecoParticleFlow.PFBlockProducer.particleFlowSimParticle_cff import *
from RecoParticleFlow.PFBlockProducer.particleFlowBlock_cff import *
from RecoParticleFlow.PFProducer.particleFlow_cff import *
from RecoTracker.IterativeTracking.iterativeTk_cff import *
#include "RecoParticleFlow/PFTracking/data/iterativeTk.cff"
#include "CalibTracker/Configuration/data/SiStrip_FakeLorentzAngle.cff"
particleFlowReco = cms.Sequence(iterTracking*caloTowersPFRec*particleFlowCluster*particleFlowTrack*particleFlowBlock*particleFlow)

