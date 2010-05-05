def customise(process):

    # add particle flow local reconstruction
    process.load("RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff")
    process.localReco += process.particleFlowCluster

    # run tracker-driven electron seeds with heavy-ion tracks
    process.load("TrackingTools.GsfTracking.FwdAnalyticalPropagator_cfi")
    process.load("RecoParticleFlow.PFTracking.trackerDrivenElectronSeeds_cff")
    process.trackerDrivenElectronSeeds.UseQuality = cms.bool(False)
    process.trackerDrivenElectronSeeds.TkColList = cms.VInputTag("hiSelectedTracks")
    process.trackerDrivenElectronSeeds.ProducePreId = cms.untracked.bool(False)
    process.trackerDrivenElectronSeeds.DisablePreId = cms.untracked.bool(True)

    # run a trimmed down PF sequence with heavy-ion vertex, no electrons, etc.
    process.load("RecoParticleFlow.Configuration.RecoParticleFlow_cff")
    process.particleFlowBlock.useConvBremPFRecTracks = cms.bool(False)
    process.particleFlowBlock.usePFatHLT = cms.bool(True)
    process.particleFlow.vertexCollection = cms.InputTag("hiSelectedVertex")
    process.particleFlow.usePFElectrons = cms.bool(False)
    process.particleFlowReco.remove(process.particleFlowTrack)
    process.particleFlowReco.remove(process.pfElectronTranslatorSequence)

    # define new high-level RECO sequence and add to top-level sequence
    process.load("RecoJets.Configuration.RecoPFJets_cff")
    process.highLevelRecoPbPb = cms.Sequence(process.trackerDrivenElectronSeeds
                                             * process.particleFlowReco
                                             * process.recoPFJets)
    process.reconstructionHeavyIons *= process.highLevelRecoPbPb

    # remove very slow jet algos
    process.globalRecoPbPb.remove(process.akPu5CaloJets)
    process.globalRecoPbPb.remove(process.akPu7CaloJets)
    
    return process
