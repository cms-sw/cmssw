def customise(process):

    # add particle flow local reconstruction
    process.load("RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff")
    process.localReco += process.particleFlowCluster

    # avoid clustering in forward regions for dramatic timing improvement 
    process.particleFlowClusterPS.thresh_Pt_Seed_Endcap = cms.double(99999.)
    process.particleFlowClusterHFEM.thresh_Pt_Seed_Endcap = cms.double(99999.)
    process.particleFlowClusterHFHAD.thresh_Pt_Seed_Endcap = cms.double(99999.)

    # run tracker-driven electron seeds with heavy-ion tracks
    process.load("TrackingTools.GsfTracking.FwdAnalyticalPropagator_cfi")
    process.load("RecoParticleFlow.PFTracking.trackerDrivenElectronSeeds_cff")
    process.trackerDrivenElectronSeeds.UseQuality = cms.bool(True)
    process.trackerDrivenElectronSeeds.TrackQuality = cms.string('highPurity')
    process.trackerDrivenElectronSeeds.TkColList = cms.VInputTag("hiSelectedTracks")
    process.trackerDrivenElectronSeeds.ProducePreId = cms.untracked.bool(False)
    process.trackerDrivenElectronSeeds.DisablePreId = cms.untracked.bool(True)

    # run a trimmed down PF sequence with heavy-ion vertex, no electrons, etc.
    process.load("RecoParticleFlow.Configuration.RecoParticleFlow_cff")
    process.particleFlowBlock.useConvBremPFRecTracks = cms.bool(False)
    process.particleFlowBlock.usePFatHLT = cms.bool(True)
    process.particleFlowBlock.useIterTracking = cms.bool(False)
    process.particleFlow.vertexCollection = cms.InputTag("hiSelectedVertex")
    process.particleFlow.usePFElectrons = cms.bool(False)
    #process.particleFlowReco.remove(process.particleFlowTrack)
    process.particleFlowReco.remove(process.particleFlowTrackWithDisplacedVertex)
    process.particleFlowReco.remove(process.pfElectronTranslatorSequence)

    # define new high-level RECO sequence and add to top-level sequence
    process.load("RecoJets.Configuration.RecoPFJets_cff")
    process.highLevelRecoPbPb = cms.Sequence(process.trackerDrivenElectronSeeds
                                             * process.particleFlowReco
                                             * process.recoPFJets)
    process.reconstructionHeavyIons *= process.highLevelRecoPbPb

    return process
