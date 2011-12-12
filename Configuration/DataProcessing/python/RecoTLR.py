import FWCore.ParameterSet.Config as cms

def customiseCommon(process):
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

    ##Ecal time bias correction
    process.ecalGlobalUncalibRecHit.doEBtimeCorrection = True
    process.ecalGlobalUncalibRecHit.doEEtimeCorrection = True
    
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
    print "WARNING"
    print "this method is outdated, please use RecoTLR.customisePPData"
    process= customisePPData(process)
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
    #add the lumi producer in the prompt reco only configuration
    process.reconstruction_step+=process.lumiProducer
    return process

##############################################################################
##############################################################################

def customiseCommonHI(process):
    
    ###############################################################################################
    ####
    ####  Top level replaces for handling strange scenarios of early HI collisions
    ####

    ## Offline Silicon Tracker Zero Suppression
    process.siStripZeroSuppression.Algorithms.CommonModeNoiseSubtractionMode = cms.string("IteratedMedian")
    process.siStripZeroSuppression.Algorithms.CutToAvoidSignal = cms.double(2.0)
    process.siStripZeroSuppression.Algorithms.Iterations = cms.int32(3)
    process.siStripZeroSuppression.storeCM = cms.bool(True)


    ###
    ###  end of top level replacements
    ###
    ###############################################################################################

    return process

##############################################################################
def customiseExpressHI(process):
    process= customiseCommonHI(process)

    import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
    process.offlineBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()
    
    return process

##############################################################################
def customisePromptHI(process):
    process= customiseCommonHI(process)

    import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
    process.offlineBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()
    
    return process

##############################################################################

def planBTracking(process):

    # stuff from LowPtTripletStep_cff
    process.lowPtTripletStepSeeds.RegionFactoryPSet.RegionPSet.ptMin=0.3

    # stuff from PixelLessStep_cff
    process.pixelLessStepClusters.oldClusterRemovalInfo=cms.InputTag("tobTecStepClusters")
    process.pixelLessStepClusters.trajectories= cms.InputTag("tobTecStepTracks")
    process.pixelLessStepClusters.overrideTrkQuals=cms.InputTag('tobTecStepSelector','tobTecStep')
    process.pixelLessStepSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.7
    process.pixelLessStepSeeds.RegionFactoryPSet.RegionPSet.originRadius = 1.5

    # stuff from PixelPairStep_cff
    process.pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.6

    # stuff from TobTecStep_cff
    process.tobTecStepClusters.oldClusterRemovalInfo=cms.InputTag("detachedTripletStepClusters")
    process.tobTecStepClusters.trajectories= cms.InputTag("detachedTripletStepTracks")
    process.tobTecStepClusters.overrideTrkQuals=cms.InputTag('detachedTripletStep')
    process.tobTecStepSeeds.RegionFactoryPSet.RegionPSet.originRadius = 5.0

    # stuff from DetachedTripletStep_cff
    process.detachedTripletStepSeeds.RegionFactoryPSet.RegionPSet.ptMin=0.35

    # stuff from iterativeTk_cff
    process.iterTracking = cms.Sequence(process.InitialStep*
                                        process.LowPtTripletStep*
                                        process.PixelPairStep*
                                        process.DetachedTripletStep*
                                        process.TobTecStep*
                                        process.PixelLessStep*
                                        process.generalTracks*
                                        process.ConvStep*
                                        process.conversionStepTracks
                                        )
    
    
    # stuff from RecoTracker_cff
    process.newCombinedSeeds.seedCollections=cms.VInputTag(
        cms.InputTag('initialStepSeeds'),
        cms.InputTag('pixelPairStepSeeds'),
    #    cms.InputTag('mixedTripletStepSeeds'),
        cms.InputTag('pixelLessStepSeeds')
        )

    # stuff from Kevin's fragment
    process.generalTracks.TrackProducers = (cms.InputTag('initialStepTracks'),
                                            cms.InputTag('lowPtTripletStepTracks'),
                                            cms.InputTag('pixelPairStepTracks'),
                                            cms.InputTag('detachedTripletStepTracks'),
                                            cms.InputTag('pixelLessStepTracks'),
                                            cms.InputTag('tobTecStepTracks'))
    process.generalTracks.hasSelector=cms.vint32(1,1,1,1,1,1)
    process.generalTracks.selectedTrackQuals = cms.VInputTag(cms.InputTag("initialStepSelector","initialStep"),
                                                             cms.InputTag("lowPtTripletStepSelector","lowPtTripletStep"),
                                                             cms.InputTag("pixelPairStepSelector","pixelPairStep"),
                                                             cms.InputTag("detachedTripletStep"),
                                                             cms.InputTag("pixelLessStepSelector","pixelLessStep"),
                                                             cms.InputTag("tobTecStepSelector","tobTecStep")
                                                             )
    process.generalTracks.setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3,4,5), pQual=cms.bool(True) ) )
    
    
    return process
