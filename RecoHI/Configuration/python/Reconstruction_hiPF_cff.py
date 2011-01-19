import FWCore.ParameterSet.Config as cms

# include  particle flow local reconstruction
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import *

# avoid clustering in forward regions for dramatic timing improvement 
particleFlowClusterPS.thresh_Pt_Seed_Endcap = cms.double(99999.)
particleFlowClusterHFEM.thresh_Pt_Seed_Endcap = cms.double(99999.)
particleFlowClusterHFHAD.thresh_Pt_Seed_Endcap = cms.double(99999.)

from RecoParticleFlow.PFTracking.pfTrack_cfi import *   
pfTrack.UseQuality = cms.bool(True)     
pfTrack.TrackQuality = cms.string('highPurity')   
pfTrack.TkColList = cms.VInputTag("hiSelectedTracks")  
pfTrack.GsfTracksInEvents = cms.bool(False)  

# run a trimmed down PF sequence with heavy-ion vertex, no electrons, etc.
from RecoParticleFlow.Configuration.RecoParticleFlow_cff import *
particleFlowBlock.useConvBremPFRecTracks = cms.bool(False)
particleFlowBlock.usePFatHLT = cms.bool(True)
particleFlowBlock.useIterTracking = cms.bool(False)
particleFlowBlock.useNuclear = cms.bool(False)   
particleFlow.vertexCollection = cms.InputTag("hiSelectedVertex")
particleFlow.usePFElectrons = cms.bool(False)
particleFlowReco.remove(particleFlowTrackWithDisplacedVertex)
particleFlowReco.remove(pfElectronTranslatorSequence)

# define new high-level RECO sequence
from RecoJets.Configuration.RecoPFJets_cff import *
HiParticleFlowReco = cms.Sequence(particleFlowCluster
                                  * pfTrack
                                  * particleFlowReco
                                  * recoPFJets)

