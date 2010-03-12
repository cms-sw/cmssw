
def customise(process):
    #####################################################################################################
    ####
    ####  Top level replaces for handling strange scenarios of early collisions
    ####

    ## TRACKING:
    ## Skip events with HV off
    process.newSeedFromTriplets.ClusterCheckPSet.MaxNumberOfPixelClusters=2000
    process.newSeedFromPairs.ClusterCheckPSet.MaxNumberOfCosmicClusters=10000
    process.secTriplets.ClusterCheckPSet.MaxNumberOfPixelClusters=1000
    process.fifthSeeds.ClusterCheckPSet.MaxNumberOfCosmicClusters = 5000
    process.fourthPLSeeds.ClusterCheckPSet.MaxNumberOfCosmicClusters=10000
    
    ###### FIXES TRIPLETS FOR LARGE BS DISPLACEMENT ######

    ### pixelTracks
    #---- replaces ----
    process.pixelTracks.RegionFactoryPSet.ComponentName = 'GlobalRegionProducerFromBeamSpot' # was GlobalRegionProducer
    process.pixelTracks.OrderedHitsFactoryPSet.GeneratorPSet.useFixedPreFiltering = True     # was False
    #---- new parameters ----
    process.pixelTracks.RegionFactoryPSet.RegionPSet.nSigmaZ  = cms.double(4.06) # was originHalfLength = 15.9; translated assuming sigmaZ ~ 3.8
    process.pixelTracks.RegionFactoryPSet.RegionPSet.beamSpot = cms.InputTag("offlineBeamSpot")

    ### 0th step of iterative tracking
    #---- replaces ----
    process.newSeedFromTriplets.RegionFactoryPSet.ComponentName = 'GlobalRegionProducerFromBeamSpot' # was GlobalRegionProducer
    process.newSeedFromTriplets.OrderedHitsFactoryPSet.GeneratorPSet.useFixedPreFiltering = True     # was False
    #---- new parameters ----
    process.newSeedFromTriplets.RegionFactoryPSet.RegionPSet.nSigmaZ   = cms.double(4.06)  # was originHalfLength = 15.9; translated assuming sigmaZ ~ 3.8
    process.newSeedFromTriplets.RegionFactoryPSet.RegionPSet.beamSpot = cms.InputTag("offlineBeamSpot")
    
    ### 2nd step of iterative tracking
    #---- replaces ----
    process.secTriplets.RegionFactoryPSet.ComponentName = 'GlobalRegionProducerFromBeamSpot' # was GlobalRegionProducer
    process.secTriplets.OrderedHitsFactoryPSet.GeneratorPSet.useFixedPreFiltering = True     # was False
    #---- new parameters ----
    process.secTriplets.RegionFactoryPSet.RegionPSet.nSigmaZ  = cms.double(4.47)  # was originHalfLength = 17.5; translated assuming sigmaZ ~ 3.8
    process.secTriplets.RegionFactoryPSet.RegionPSet.beamSpot = cms.InputTag("offlineBeamSpot")
    
    ## Primary Vertex
    process.offlinePrimaryVerticesWithBS.PVSelParameters.maxDistanceToBeam = 2
    process.offlinePrimaryVerticesWithBS.TkFilterParameters.maxNormalizedChi2 = 20
    process.offlinePrimaryVerticesWithBS.TkFilterParameters.minSiliconHits = 6
    process.offlinePrimaryVerticesWithBS.TkFilterParameters.maxD0Significance = 100
    process.offlinePrimaryVerticesWithBS.TkFilterParameters.minPixelHits = 1
    process.offlinePrimaryVerticesWithBS.TkClusParameters.zSeparation = 10
    process.offlinePrimaryVertices.PVSelParameters.maxDistanceToBeam = 2
    process.offlinePrimaryVertices.TkFilterParameters.maxNormalizedChi2 = 20
    process.offlinePrimaryVertices.TkFilterParameters.minSiliconHits = 6
    process.offlinePrimaryVertices.TkFilterParameters.maxD0Significance = 100
    process.offlinePrimaryVertices.TkFilterParameters.minPixelHits = 1
    process.offlinePrimaryVertices.TkClusParameters.zSeparation = 10
    
    ## ECAL 
    process.ecalRecHit.ChannelStatusToBeExcluded = [ 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 78, 142 ]
    
    
    ## HCAL temporary fixes
    process.hfreco.firstSample  = 1
    process.hfreco.samplesToAdd = 4
    
    process.zdcreco.firstSample = 4
    process.zdcreco.samplesToAdd = 3
    
    ## EGAMMA
    process.ecalDrivenElectronSeeds.SCEtCut = cms.double(1.0)
    process.ecalDrivenElectronSeeds.applyHOverECut = cms.bool(False)
    process.ecalDrivenElectronSeeds.SeedConfiguration.z2MinB = cms.double(-0.9)
    process.ecalDrivenElectronSeeds.SeedConfiguration.z2MaxB = cms.double(0.9)
    process.ecalDrivenElectronSeeds.SeedConfiguration.r2MinF = cms.double(-1.5)
    process.ecalDrivenElectronSeeds.SeedConfiguration.r2MaxF = cms.double(1.5)
    process.ecalDrivenElectronSeeds.SeedConfiguration.rMinI = cms.double(-2.)
    process.ecalDrivenElectronSeeds.SeedConfiguration.rMaxI = cms.double(2.)
    process.ecalDrivenElectronSeeds.SeedConfiguration.DeltaPhi1Low = cms.double(0.3)
    process.ecalDrivenElectronSeeds.SeedConfiguration.DeltaPhi1High = cms.double(0.3)
    process.ecalDrivenElectronSeeds.SeedConfiguration.DeltaPhi2 = cms.double(0.3)
    process.gsfElectrons.applyPreselection = cms.bool(False)
    process.photons.minSCEtBarrel = 1.
    process.photons.minSCEtEndcap =1.
    process.photonCore.minSCEt = 1.
    process.conversionTrackCandidates.minSCEt =1.
    process.conversions.minSCEt =1.
    process.trackerOnlyConversions.AllowTrackBC = cms.bool(False)
    process.trackerOnlyConversions.AllowRightBC = cms.bool(False)
    process.trackerOnlyConversions.MinApproach = cms.double(-.25)
    process.trackerOnlyConversions.DeltaCotTheta = cms.double(.07)
    process.trackerOnlyConversions.DeltaPhi = cms.double(.2)
    
    ###
    ###  end of top level replacements
    ###
    ###############################################################################################

    return (process)
