import FWCore.ParameterSet.Config as cms

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
    if hasattr(process,'CSCGeometryESModule'):
        process.CSCGeometryESModule.useGangedStripsInME1a = cms.bool(False)
    if hasattr(process,'CSCIndexerESProducer'):
        process.CSCIndexerESProducer.AlgoName=cms.string("CSCIndexerPostls1")
    if hasattr(process,'CSCChannelMapperESProducer'):
        process.CSCChannelMapperESProducer.AlgoName=cms.string("CSCChannelMapperPostls1")
    if hasattr(process,'csc2DRecHits'):
        process.csc2DRecHits.readBadChannels = cms.bool(False)
        process.csc2DRecHits.CSCUseGasGainCorrections = cms.bool(False)
    if hasattr(process,'valCscTriggerPrimitiveDigis'):
        #this is not doing anything at the moment
        process.valCscTriggerPrimitiveDigis.commonParam.gangedME1a = cms.untracked.bool(False)
    if hasattr(process,'valCsctfTrackDigis'):
        process.valCsctfTrackDigis.gangedME1a = cms.untracked.bool(False)

    from SLHCUpgradeSimulations.Configuration.postLS1Customs import customise_Reco,customise_RawToDigi,customise_DQM
    if hasattr(process,'RawToDigi'):
        process=customise_RawToDigi(process)
    if hasattr(process,'reconstruction'):
        process=customise_Reco(process)
    if hasattr(process,'dqmoffline_step'):
        process=customise_DQM(process)

    return process

##############################################################################
# common+ "25ns" Use this for data daking starting from runs in 2015C (>= 253256 )
def customiseDataRun2Common_25ns(process):
    process = customiseDataRun2Common(process)

    from L1Trigger.L1TCommon.customsPostLS1 import customiseL1RecoForStage1
    process=customiseL1RecoForStage1(process)

    from SLHCUpgradeSimulations.Configuration.postLS1Customs import customise_DQM_25ns
    if hasattr(process,'dqmoffline_step'):
        process=customise_DQM_25ns(process)
    return process

# common+50ns. Needed only for runs >= 253000 if taken with 50ns
def customiseDataRun2Common_50nsRunsAfter253000(process):
    process = customiseDataRun2Common(process)

    from L1Trigger.L1TCommon.customsPostLS1 import customiseL1RecoForStage1
    process=customiseL1RecoForStage1(process)

    if hasattr(process,'particleFlowClusterECAL'):
        process.particleFlowClusterECAL.energyCorrector.autoDetectBunchSpacing = False
        process.particleFlowClusterECAL.energyCorrector.bunchSpacing = cms.int32(50)
    if hasattr(process,'ecalMultiFitUncalibRecHit'):
        process.ecalMultiFitUncalibRecHit.algoPSet.useLumiInfoRunHeader = False
        process.ecalMultiFitUncalibRecHit.algoPSet.bunchSpacing = cms.int32(50)

    return process

##############################################################################
def customiseCosmicDataRun2(process):
    process = customiseCosmicData(process)
    process = customiseDataRun2Common_25ns(process)
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

    from RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi import onlineBeamSpotProducer
    process.offlineBeamSpot = onlineBeamSpotProducer.clone()
    
    return process

##############################################################################
def customiseExpressRun2(process):
    process = customiseExpress(process)
    process = customiseDataRun2Common_25ns(process)
    return process

def customiseExpressRun2_50ns(process):
    process = customiseExpress(process)
    process = customiseDataRun2Common_50nsRunsAfter253000(process)
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
        from RecoLuminosity.LumiProducer.lumiProducer_cff import lumiProducer,LumiDBService
        process.lumiProducer=lumiProducer
    process.reconstruction_step+=process.lumiProducer

    return process

##############################################################################
def customisePromptRun2(process):
    process = customisePrompt(process)
    process = customiseDataRun2Common_25ns(process)
    return process

def customisePromptRun2_50ns(process):
    process = customisePrompt(process)
    process = customiseDataRun2Common_50nsRunsAfter253000(process)
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

    from RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi import onlineBeamSpotProducer
    process.offlineBeamSpot = onlineBeamSpotProducer.clone()
    
    return process

##############################################################################
def customisePromptHI(process):
    #deprecated process= customiseCommonHI(process)

    from RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi import onlineBeamSpotProducer
    process.offlineBeamSpot = onlineBeamSpotProducer.clone()

     #add the lumi producer in the prompt reco only configuration
    if not hasattr(process,'lumiProducer'):
        #unscheduled..
        from RecoLuminosity.LumiProducer.lumiProducer_cff import lumiProducer,LumiDBService
        process.lumiProducer=lumiProducer
    process.reconstruction_step+=process.lumiProducer
        

    return process

##############################################################################
