import FWCore.ParameterSet.Config as cms

def customiseCommon(process):
    
    #####################################################################################################
    ####
    ####  Top level replaces for handling strange scenarios of early collisions
    ####

    ## TRACKING:
    ## Skip events with HV off
    process.newSeedFromTriplets.ClusterCheckPSet.MaxNumberOfPixelClusters=2000
    process.newSeedFromPairs.ClusterCheckPSet.MaxNumberOfStripClusters=20000
    process.secTriplets.ClusterCheckPSet.MaxNumberOfPixelClusters=2000
    process.fifthSeeds.ClusterCheckPSet.MaxNumberOfStripClusters = 20000
    process.fourthPLSeeds.ClusterCheckPSet.MaxNumberOfStripClusters=20000
    process.thTripletsA.ClusterCheckPSet.MaxNumberOfPixelClusters = 5000
    process.thTripletsB.ClusterCheckPSet.MaxNumberOfPixelClusters = 5000

    ###### FIXES TRIPLETS FOR LARGE BS DISPLACEMENT ######

    ### prevent bias in pixel vertex
    process.pixelVertices.useBeamConstraint = False
    
    ### pixelTracks
    #---- new parameters ----
    process.pixelTracks.RegionFactoryPSet.RegionPSet.nSigmaZ  = 4.06
    process.pixelTracks.RegionFactoryPSet.RegionPSet.originHalfLength = cms.double(40.6)

    ### 0th step of iterative tracking
    #---- new parameters ----
    process.newSeedFromTriplets.RegionFactoryPSet.RegionPSet.nSigmaZ   = cms.double(4.06)  
    process.newSeedFromTriplets.RegionFactoryPSet.RegionPSet.originHalfLength = 40.6

    ### 2nd step of iterative tracking
    #---- new parameters ----
    process.secTriplets.RegionFactoryPSet.RegionPSet.nSigmaZ  = cms.double(4.47)  
    process.secTriplets.RegionFactoryPSet.RegionPSet.originHalfLength = 44.7

    ## ECAL 
    process.ecalRecHit.ChannelStatusToBeExcluded = [ 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 78, 142 ]

    ###
    ###  end of top level replacements
    ###
    ###############################################################################################

    return (process)


##############################################################################
def customisePPData(process):
    process= customiseCommon(process)

    ## particle flow HF cleaning
    process.particleFlowRecHitHCAL.LongShortFibre_Cut = 30.
    process.particleFlowRecHitHCAL.ApplyPulseDPG = True

    ## HF cleaning for data only
    process.hcalRecAlgos.SeverityLevels[3].RecHitFlags.remove("HFDigiTime")
    process.hcalRecAlgos.SeverityLevels[4].RecHitFlags.append("HFDigiTime")

    ##beam-halo-id for data only
    process.CSCHaloData.ExpectedBX = cms.int32(3)

    ## hcal hit flagging
    process.hfreco.PETstat.flagsToSkip  = 2
    process.hfreco.S8S1stat.flagsToSkip = 18
    process.hfreco.S9S1stat.flagsToSkip = 26
    
    return process


##############################################################################
def customisePPMC(process):
    process=customiseCommon(process)
    
    return process

##############################################################################
def customiseCosmicData(process):

    return process

##############################################################################
def customiseCosmicMC(process):
    
    return process
        
##############################################################################
def customiseVALSKIM(process):
    process= customisePPData(process)
    process.reconstruction.remove(process.lumiProducer)
    return process
                
##############################################################################
def customiseExpress(process):
    process= customisePPData(process)

    import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
    process.offlineBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()
    
    return process

##############################################################################
def customisePrompt(process):
    process= customisePPData(process)

    import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
    process.offlineBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()
    
    return process
