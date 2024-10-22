import FWCore.ParameterSet.Config as cms

def customiseCommon(process):

    
    #####################################################################################################
    ####
    ####  Top level replaces for handling strange scenarios of early collisions
    ####

    ## TRACKING:
    ## Skip events with HV off
    process.newSeedFromTriplets.ClusterCheckPSet.MaxNumberOfPixelClusters=2000
    process.newSeedFromPairs.ClusterCheckPSet.MaxNumberOfStripClusters=10000
    process.secTriplets.ClusterCheckPSet.MaxNumberOfPixelClusters=2000
    process.fifthSeeds.ClusterCheckPSet.MaxNumberOfStripClusters = 10000
    process.fourthPLSeeds.ClusterCheckPSet.MaxNumberOfStripClusters=10000

    ###### FIXES TRIPLETS FOR LARGE BS DISPLACEMENT ######

    ### prevent bias in pixel vertex
    process.pixelVertices.useBeamConstraint = False
    
    ### pixelTracks
    #---- new parameters ----
    process.pixelTracks.RegionFactoryPSet.RegionPSet.nSigmaZ  = cms.double(4.06) # was originHalfLength = 15.9; translated assuming sigmaZ ~ 3.8
    
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
    process.offlinePrimaryVerticesWithBS.TkClusParameters.zSeparation = 1
    process.offlinePrimaryVertices.PVSelParameters.maxDistanceToBeam = 2
    process.offlinePrimaryVertices.TkFilterParameters.maxNormalizedChi2 = 20
    process.offlinePrimaryVertices.TkFilterParameters.minSiliconHits = 6
    process.offlinePrimaryVertices.TkFilterParameters.maxD0Significance = 100
    process.offlinePrimaryVertices.TkFilterParameters.minPixelHits = 1
    process.offlinePrimaryVertices.TkClusParameters.zSeparation = 1

    ## ECAL 
    process.ecalRecHit.ChannelStatusToBeExcluded = [ 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 78, 142 ]

    ## HCAL temporary fixes
    process.hfreco.samplesToAdd = 4
    
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

    return (process)


##############################################################################
def customisePPData(process):
    process= customiseCommon(process)
    process.hfreco.firstSample=3
    ##Preshower
    process.ecalPreshowerRecHit.ESBaseline = 0

    ##Preshower algo for data is different than for MC
    process.ecalPreshowerRecHit.ESRecoAlgo = cms.untracked.int32(1)

    return process


##############################################################################
def customisePPMC(process):
    process=customiseCommon(process)
    process.hfreco.firstSample=1
    
    return process

##############################################################################
def customiseCosmicData(process):
    process.ecalPreshowerRecHit.ESBaseline = cms.int32(0)
    process.ecalPreshowerRecHit.ESRecoAlgo = cms.untracked.int32(1)
    
    return process

##############################################################################
def customiseCosmicMC(process):
    
    return process
        

##############################################################################
def customiseExpress(process):
    process= customisePPData(process)

    return process
