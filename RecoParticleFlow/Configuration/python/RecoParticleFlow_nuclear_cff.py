# The following comments couldn't be translated into the new config version:

#include "CalibTracker/Configuration/data/SiStrip_FakeLorentzAngle.cff"

import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.towerMakerPF_cff import *
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import *
from RecoParticleFlow.PFTracking.particleFlowTrack_cff import *
#include "RecoParticleFlow/PFTracking/data/nuclear.cff" 
from RecoParticleFlow.PFTracking.nuclearRemainingHits_cff import *
from RecoParticleFlow.PFBlockProducer.particleFlowSimParticle_cff import *
from RecoParticleFlow.PFBlockProducer.particleFlowBlock_cff import *
from RecoParticleFlow.PFProducer.particleFlow_cff import *
from RecoParticleFlow.PFTracking.iterativeTk_cff import *
particleFlowRecoNuclear = cms.Sequence(iterativeTk*nuclearRemainingHits*caloTowersPFRec*particleFlowCluster*particleFlowTrackWithNuclear*particleFlowBlock*particleFlow)
particleFlowBlock.useNuclear = True

