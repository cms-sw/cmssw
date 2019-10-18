import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFTracking.particleFlowTrack_cff import *
from RecoParticleFlow.PFTracking.trackerDrivenElectronSeeds_cfi import *
from RecoParticleFlow.PFProducer.particleFlowSimParticle_cff import *
from RecoParticleFlow.PFProducer.particleFlowBlock_cff import *
from RecoParticleFlow.PFProducer.particleFlow_cff import *
from RecoTracker.Configuration.RecoTracker_cff import *

import Geometry.CaloEventSetup.caloTowerConstituents_cfi

CaloTowerConstituentsMapBuilder = Geometry.CaloEventSetup.caloTowerConstituents_cfi.caloTowerConstituents.clone()
CaloTowerConstituentsMapBuilder.MapFile = "Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz"

particleFlowRecoTask = cms.Task(
    ckftracksTask,
    trackerDrivenElectronSeeds,
    particleFlowBlock,
    particleFlowTmp
    )
particleFlowReco = cms.Sequence(particleFlowRecoTask)
