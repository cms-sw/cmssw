import FWCore.ParameterSet.Config as cms

def customise(process):

    # Customize process to run PF *without* electrons

    # add particle flow local reconstruction
    process.load("RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff")
    process.localReco += process.particleFlowCluster

    process.load("RecoParticleFlow.PFTracking.pfTrack_cfi")
    process.pfTrack.UseQuality = cms.bool(True)   
    process.pfTrack.TrackQuality = cms.string('highPurity')   
    process.pfTrack.TkColList = cms.VInputTag("hiGeneralTracks")  
    process.pfTrack.PrimaryVertexLabel = cms.InputTag("hiSelectedVertex")
    process.pfTrack.MuColl = cms.InputTag("hiMuons1stStep")
    process.pfTrack.GsfTracksInEvents = cms.bool(False)
    
    # run a trimmed down PF sequence with heavy-ion vertex, no conversions, nucl int, etc.
    process.load("RecoParticleFlow.Configuration.RecoParticleFlow_cff")

    process.particleFlowBlock.useConvBremPFRecTracks = cms.bool(False)
    process.particleFlowBlock.useIterTracking = cms.bool(False)
    process.particleFlowBlock.useNuclear = cms.bool(False)
    process.particleFlowBlock.useConversions = cms.bool(False)

    process.particleFlowTmp.vertexCollection = cms.InputTag("hiSelectedVertex")
    process.particleFlowTmp.usePFElectrons = cms.bool(False)
    process.particleFlowTmp.muons = cms.InputTag("hiMuons1stStep")
    process.particleFlowTmp.usePFConversions = cms.bool(False)

    process.electronsCiCLoose.verticesCollection = cms.InputTag("hiSelectedVertex")

    # define new high-level RECO sequence and add to top-level sequence
    process.highLevelRecoPbPb = cms.Sequence(process.pfTrack
                                             * process.pfGsfElectronCiCSelectionSequence
                                             * process.particleFlowBlock
                                             * process.particleFlowTmp
                                             )
    
    process.reconstructionHeavyIons *= process.highLevelRecoPbPb
    
    return process
