from __future__ import print_function
import FWCore.ParameterSet.Config as cms

##############################################################################
# common utilities
##############################################################################
def _swapOfflineBSwithOnline(process):
    from RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi import onlineBeamSpotProducer
    process.offlineBeamSpot = onlineBeamSpotProducer.clone()
    return process

def _overridesFor50ns(process):
    process.bunchSpacingProducer.bunchSpacingOverride = cms.uint32(50)
    process.bunchSpacingProducer.overrideBunchSpacing = cms.bool(True)
    
    return process

##############################################################################
# post-era customizations
# these are here instead of generating Data-specific eras
##############################################################################

def customisePostEra_Run2_25ns(process):
    return process

def customisePostEra_Run2_2016(process):
    return process

def customisePostEra_Run2_2017(process):
    return process

def customisePostEra_Run2_2017_express_trackingOnly(process):
    customisePostEra_Run2_2017(process)
    from Calibration.TkAlCaRecoProducers.PCLHPbeamspot_custom import customise_HPbeamspot as _customise_HPbeamspot
    _customise_HPbeamspot(process)
    return process

def customisePostEra_Run2_2017_harvesting_trackingOnly(process):
    from Calibration.TkAlCaRecoProducers.PCLHPbeamspot_custom import customise_HPbeamspot as _customise_HPbeamspot
    _customise_HPbeamspot(process)
    return process

def customisePostEra_Run2_2017_pp_on_XeXe(process):
    customisePostEra_Run2_2017(process)
    return process

def customisePostEra_Run2_2017_ppRef(process):
    customisePostEra_Run2_2017(process)
    return process

# 2018 equivalents
def customisePostEra_Run2_2018(process):
    #start with a repeat of 2017
    customisePostEra_Run2_2017(process)
    return process

def customisePostEra_Run2_2018_express_trackingOnly(process):
    #start with a repeat of 2017
    customisePostEra_Run2_2017_express_trackingOnly(process)
    return process

def customisePostEra_Run2_2018_harvesting_trackingOnly(process):
    #start with a repeat of 2017
    customisePostEra_Run2_2017_harvesting_trackingOnly(process)
    return process

def customise_HI_PostEra_Run2_2018(process):
    customisePostEra_Run2_2018(process)
    return process

def customisePostEra_Run2_2018_pp_on_AA(process):
    customisePostEra_Run2_2018(process)
    return process

def customisePostEra_Run2_2018_pp_on_AA_express_trackingOnly(process):
    customisePostEra_Run2_2018_express_trackingOnly(process)
    from DQM.TrackingMonitorSource.PPonAATrackingOnly_custom import customise_PPonAATrackingOnlyDQM as _customise_PPonAATrackingOnlyDQM
    _customise_PPonAATrackingOnlyDQM(process)
    return process

# Run3 equivalents

def customisePostEra_Run3(process):
    #start with a repeat of 2018
    customisePostEra_Run2_2018(process)
    return process

def customisePostEra_Run3_express_trackingOnly(process):
    #start with a repeat of 2018
    customisePostEra_Run2_2018_express_trackingOnly(process)
    return process

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
def customiseCosmicMC(process):
    return process
        
##############################################################################
def customiseVALSKIM(process):
    print("WARNING")
    print("this method is outdated, please use RecoTLR.customisePPData")
    process= customisePPData(process)
    return process

                
##############################################################################
def customiseExpress(process):
    process= customisePPData(process)
    process = _swapOfflineBSwithOnline(process)
    return process

##############################################################################
def customisePrompt(process):
    process= customisePPData(process)

    return process

##############################################################################
# Heavy Ions
##############################################################################
# keep it in case modification is needed
def customiseCommonHI(process):
    return process

##############################################################################
def customiseExpressHI(process):
    process = customiseCommonHI(process)
    process = _swapOfflineBSwithOnline(process)
    
    return process

##############################################################################
def customisePromptHI(process):
    process = customiseCommonHI(process)

    return process

##############################################################################
##############################################################################
##
##  ALL FUNCTIONS BELOW ARE GOING TO BE REMOVED IN 81X
##
##############################################################################
##############################################################################
# this is supposed to be added on top of other (Run1) data customs
def customiseDataRun2Common(process):
    from SLHCUpgradeSimulations.Configuration.muonCustoms import unganged_me1a_geometry,customise_csc_LocalReco
    process = unganged_me1a_geometry(process)
    process = customise_csc_LocalReco(process)

    if hasattr(process,'valCscTriggerPrimitiveDigis'):
        #this is not doing anything at the moment
        process.valCscTriggerPrimitiveDigis.commonParam.gangedME1a = cms.bool(False)
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

# add stage1
def customiseDataRun2Common_withStage1(process):
    process = customiseDataRun2Common(process)

    from L1Trigger.L1TCommon.customsPostLS1 import customiseL1RecoForStage1
    process=customiseL1RecoForStage1(process)

    return process 

##############################################################################
# common+ "25ns" Use this for data daking starting from runs in 2015C (>= 253256 )
def customiseDataRun2Common_25ns(process):
    process = customiseDataRun2Common_withStage1(process)

    from SLHCUpgradeSimulations.Configuration.postLS1Customs import customise_DQM_25ns
    if hasattr(process,'dqmoffline_step'):
        process=customise_DQM_25ns(process)
    return process

# common+50ns. Needed only for runs >= 253000 if taken with 50ns
def customiseDataRun2Common_50nsRunsAfter253000(process):
    process = customiseDataRun2Common_withStage1(process)

    process = _overridesFor50ns(process)

    return process

##############################################################################
# keep it in case modification is needed
def customiseRun2CommonHI(process):
    process = customiseDataRun2Common_withStage1(process)
    
    process = _overridesFor50ns(process)
    # HI Specific additional customizations:
    # from L1Trigger.L1TCommon.customsPostLS1 import customiseSimL1EmulatorForPostLS1_Additional_HI
    # process = customiseSimL1EmulatorForPostLS1_Additional_HI(process)

    return process
