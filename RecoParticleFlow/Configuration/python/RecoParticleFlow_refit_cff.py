import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFTracking.particleFlowTrack_cff import *
from RecoParticleFlow.PFTracking.trackerDrivenElectronSeeds_cfi import *
from RecoParticleFlow.PFProducer.particleFlowSimParticle_cff import *
from RecoParticleFlow.PFProducer.particleFlowBlock_cff import *
from RecoParticleFlow.PFProducer.particleFlow_cff import *
from RecoTracker.Configuration.RecoTracker_cff import *
CaloTowerConstituentsMapBuilder = cms.ESProducer("CaloTowerConstituentsMapBuilder",
    MapFile = cms.untracked.string('Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz')
)

particleFlowReco = cms.Sequence(
    ckftracks*
    trackerDrivenElectronSeeds*
    particleFlowBlock*
    particleFlowTmp
    )

