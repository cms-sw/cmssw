import FWCore.ParameterSet.Config as cms

# include  particle flow local reconstruction
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import *

# avoid clustering in forward regions for dramatic timing improvement 
particleFlowClusterPS.thresh_Pt_Seed_Endcap = cms.double(99999.)
particleFlowClusterHFEM.thresh_Pt_Seed_Endcap = cms.double(99999.)
particleFlowClusterHFHAD.thresh_Pt_Seed_Endcap = cms.double(99999.)

# run tracker-driven electron seeds with heavy-ion tracks
from TrackingTools.GsfTracking.FwdAnalyticalPropagator_cfi import *
from RecoParticleFlow.PFTracking.trackerDrivenElectronSeeds_cff import *
trackerDrivenElectronSeeds.UseQuality = cms.bool(True)
trackerDrivenElectronSeeds.TrackQuality = cms.string('highPurity')
trackerDrivenElectronSeeds.TkColList = cms.VInputTag("hiSelectedTracks")
trackerDrivenElectronSeeds.ProducePreId = cms.untracked.bool(False)
trackerDrivenElectronSeeds.DisablePreId = cms.untracked.bool(True)

# run a trimmed down PF sequence with heavy-ion vertex, no electrons, etc.
from RecoParticleFlow.Configuration.RecoParticleFlow_cff import *
particleFlowBlock.useConvBremPFRecTracks = cms.bool(False)
particleFlowBlock.usePFatHLT = cms.bool(True)
particleFlowBlock.useIterTracking = cms.bool(False)
particleFlow.vertexCollection = cms.InputTag("hiSelectedVertex")
particleFlow.usePFElectrons = cms.bool(False)
#particleFlowReco.remove(particleFlowTrack)
particleFlowReco.remove(particleFlowTrackWithDisplacedVertex)
particleFlowReco.remove(pfElectronTranslatorSequence)

# define new high-level RECO sequence
from RecoJets.Configuration.RecoPFJets_cff import *
HiParticleFlowReco = cms.Sequence(particleFlowCluster
                                  * trackerDrivenElectronSeeds
                                  * particleFlowReco
                                  * recoPFJets)

