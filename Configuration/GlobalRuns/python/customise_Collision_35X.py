
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

    ### prevent bias in pixel vertex
    process.pixelVertices.useBeamConstraint = False
    
    ### pixelTracks
    #---- replaces ----
    process.pixelTracks.RegionFactoryPSet.ComponentName = 'GlobalRegionProducerFromBeamSpot' # was GlobalRegionProducer
    #---- new parameters ----
    process.pixelTracks.RegionFactoryPSet.RegionPSet.nSigmaZ  = cms.double(4.06) # was originHalfLength = 15.9; translated assuming sigmaZ ~ 3.8
    process.pixelTracks.RegionFactoryPSet.RegionPSet.beamSpot = cms.InputTag("offlineBeamSpot")
    
    ### 0th step of iterative tracking
    #---- replaces ----
    process.newSeedFromTriplets.RegionFactoryPSet.ComponentName = 'GlobalRegionProducerFromBeamSpot' # was GlobalRegionProducer
    #---- new parameters ----
    process.newSeedFromTriplets.RegionFactoryPSet.RegionPSet.nSigmaZ   = cms.double(4.06)  # was originHalfLength = 15.9; translated assuming sigmaZ ~ 3.8
    process.newSeedFromTriplets.RegionFactoryPSet.RegionPSet.beamSpot = cms.InputTag("offlineBeamSpot")

    ### 2nd step of iterative tracking
    #---- replaces ----
    process.secTriplets.RegionFactoryPSet.ComponentName = 'GlobalRegionProducerFromBeamSpot' # was GlobalRegionProducer
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

    ##Preshower
    process.ecalPreshowerRecHit.ESBaseline = 0

    ##Preshower algo for data is different than for MC
    process.ecalPreshowerRecHit.ESRecoAlgo = cms.untracked.int32(1)

    ## HCAL temporary fixes
    process.hfreco.firstSample  = 3
    process.hfreco.samplesToAdd = 4
    
    process.zdcreco.firstSample = 4
    process.zdcreco.samplesToAdd = 3

    ## EGAMMA
    process.gsfElectrons.applyPreselection = cms.bool(False)
    process.photons.minSCEtBarrel = 2.
    process.photons.minSCEtEndcap =2.
    process.photonCore.minSCEt = 2.
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


    
    # produce L1 trigger object maps (temporary fix for HLT mistake 
    # in event content definition of RAW datatier for stream A)
    import L1Trigger.GlobalTrigger.gtDigis_cfi
    process.hltL1GtObjectMap = L1Trigger.GlobalTrigger.gtDigis_cfi.gtDigis.clone(
        GmtInputTag = cms.InputTag( "gtDigis" ),
        GctInputTag = cms.InputTag( "gctDigis" ),
        CastorInputTag = cms.InputTag( "castorL1Digis" ),
        ProduceL1GtDaqRecord = cms.bool( False ),
        ProduceL1GtEvmRecord = cms.bool( False ),
        ProduceL1GtObjectMapRecord = cms.bool( True ),
        WritePsbL1GtDaqRecord = cms.bool( False ),
        ReadTechnicalTriggerRecords = cms.bool( True ),
        EmulateBxInEvent = cms.int32( 1 ),
        AlternativeNrBxBoardDaq = cms.uint32( 0 ),
        AlternativeNrBxBoardEvm = cms.uint32( 0 ),
        BstLengthBytes = cms.int32( -1 ),
        TechnicalTriggersInputTags = cms.VInputTag( 'simBscDigis' ),
        RecordLength = cms.vint32( 3, 0 )
        )
    process.L1GtObjectMap_step = cms.Path(process.hltL1GtObjectMap)
    process.schedule.insert(process.schedule.index(process.raw2digi_step)+1,process.L1GtObjectMap_step)

    
    return (process)
