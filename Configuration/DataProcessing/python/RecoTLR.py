import FWCore.ParameterSet.Config as cms

from SLHCUpgradeSimulations.Configuration.postLS1Customs import customise_Reco,customise_RawToDigi
from RecoTracker.Configuration.customiseForRunI import customiseForRunI
#gone with the fact that there is no difference between production and development sequence
#def customiseCommon(process):
#    return (process)


##############################################################################
def customisePPData(process):
    #deprecated process= customiseCommon(process)
    ##all customisation for data are now deprecated to Reconstruction_Data_cff
    #left as a place holder to alter production sequences in case of emergencies
    return process


##############################################################################
def customisePPMC(process):
    #deprecated process=customiseCommon(process)
    #left as a place holder to alter production sequences in case of emergencies    
    return process

##############################################################################
def customiseCosmicData(process):

    return process



##############################################################################
# this is supposed to be added on top of other (Run1) data customs
def customiseDataRun2Common(process):
    process.CSCGeometryESModule.useGangedStripsInME1a = cms.bool(False)
    process.CSCIndexerESProducer.AlgoName=cms.string("CSCIndexerPostls1")
    process.CSCChannelMapperESProducer.AlgoName=cms.string("CSCChannelMapperPostls1")
    process.csc2DRecHits.readBadChannels = cms.bool(False)
    process.csc2DRecHits.CSCUseGasGainCorrections = cms.bool(False)
    if hasattr(process,'valCscTriggerPrimitiveDigis'):
        #this is not doing anything at the moment
        process.valCscTriggerPrimitiveDigis.commonParam.gangedME1a = cms.untracked.bool(False)
    if hasattr(process,'valCsctfTrackDigis'):
        process.valCsctfTrackDigis.gangedME1a = cms.untracked.bool(False)

    process=customise_Reco(process)
    process=customise_RawToDigi(process)

    return process

##############################################################################
def customiseCosmicDataRun2(process):
    process = customiseCosmicData(process)
    process = customiseDataRun2Common(process)
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
def customiseExpressRun2(process):
    process = customiseExpress(process)
    process = customiseDataRun2Common(process)
    process.SiStripClusterChargeCutTight.value = -1.
    process.SiStripClusterChargeCutLoose.value = -1.
    return process

def customiseExpressRun2B0T(process):
    process=customiseForRunI(process)
    process=customiseExpressRun2(process)
    return process

##############################################################################
def customisePrompt(process):
    process= customisePPData(process)

    #add the lumi producer in the prompt reco only configuration
    if not hasattr(process,'lumiProducer'):
        #unscheduled..
        from RecoLuminosity.LumiProducer.lumiProducer_cff import *
        process.lumiProducer=lumiProducer
    process.reconstruction_step+=process.lumiProducer

    return process

##############################################################################
def customisePromptRun2(process):
    process = customisePrompt(process)
    process = customiseDataRun2Common(process)
    process.SiStripClusterChargeCutTight.value = -1.
    process.SiStripClusterChargeCutLoose.value = -1.
    return process

def customisePromptRun2B0T(process):
    process=customiseForRunI(process)
    process=customisePromptRun2(process)
    return process


##############################################################################

#gone with the fact that there is no difference between production and development sequence
#def customiseCommonHI(process):
#    return process

##############################################################################
def customiseExpressHI(process):
    #deprecated process= customiseCommonHI(process)

    import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
    process.offlineBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()
    
    return process

##############################################################################
def customisePromptHI(process):
    #deprecated process= customiseCommonHI(process)

    import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
    process.offlineBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()

     #add the lumi producer in the prompt reco only configuration
    if not hasattr(process,'lumiProducer'):
        #unscheduled..
        from RecoLuminosity.LumiProducer.lumiProducer_cff import *
        process.lumiProducer=lumiProducer
    process.reconstruction_step+=process.lumiProducer
        

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


    if hasattr(process,'dqmoffline_step'):
        process.dqmoffline_step.remove(process.TrackMonStep4)
        #process.dqmoffline_step.remove(process.TrackMonStep5)
        
    return process
