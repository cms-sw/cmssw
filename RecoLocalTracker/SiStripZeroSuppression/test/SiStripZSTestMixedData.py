# Auto generated configuration file
# using: 
# Revision: 1.222.2.6 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: hiRecoDM -n 10 --scenario HeavyIons -s RAW2DIGI,L1Reco,RECO --processName TEST --datatier GEN-SIM-RECO --eventcontent FEVTDEBUG --customise SimGeneral.DataMixingModule.DataMixer_DataConditions_3_8_X_data2010 --cust_function customise --geometry DB --filein file:DMRawSimOnReco_DIGI2RAW.root --fileout hiRecoDM_RECO.root --conditions FrontierConditions_GlobalTag,MC_38Y_V12::All --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('TEST')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
#process.load('Configuration.StandardSequences.MixingNoPileUp_cff')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.ReconstructionHeavyIons_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContentHeavyIons_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    annotation = cms.untracked.string('hiRecoDM nevts:10'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
process.Timing = cms.Service("Timing")
#process.MessageLogger.cerr.threshold = cms.untracked.string("DEBUG")
#process.MessageLogger.categories+=cms.vstring("SiStripFedCMExtractor","SiStripProcessedRawDigiSkimProducer")

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    #'/store/express/Run2010B/ExpressPhysics/FEVT/Express-v2/000/146/417/10F981FF-5EC6-DF11-9657-0030486733B4.root'
    #'file:../testGenSimOnReco/SingleZmumu_MatchVertexDM_DIGI2RAW.root'
    #'file:DMRawSimOnReco_DIGI2RAW.root'
    #'file:DMRawSimOnReco_DIGI2RAW.root'
	'/store/data/Run2010B/HeavyIonTest/RAW/v1/000/146/421/E6B24CF0-5EC6-DF11-B52D-00304879FC6C.root'
    )
)

# Output definition
process.RECOoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    #outputCommands = process.RECOEventContent.outputCommands,
    fileName = cms.untracked.string('hiReco_E6B24CF0-5EC6-DF11-B52D-00304879FC6C_10ev.root'),
	#fileName = cms.untracked.string('test.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-RECO')
    )
)

# Additional output definition
#process.RECOoutput.outputCommands.extend(['keep *_siStripProcessedRawDigisSkim_*_*',
#                                               'keep *_*_APVCM_*'])

# Other statements
process.GlobalTag.globaltag = 'MC_39Y_V4::All'
#process.GlobalTag.globaltag = 'MC_38Y_V12::All'
#process.GlobalTag.globaltag = 'START38_V8::All'
#process.GlobalTag.globaltag = 'GR_R_38X_V7::All'
#process.siStripDigis.ProductLabel = "source"
#process.siStripZeroSuppression.storeCM = cms.bool(True)
#process.siStripZeroSuppression.produceRawDigis = cms.bool(True)
#process.siStripZeroSuppression.produceCalculatedBaseline = cms.bool(True)
#process.siStripZeroSuppression.produceBaselinePoints = cms.bool(False)

## Offline Silicon Tracker Zero Suppression
process.siStripZeroSuppression.Algorithms.PedestalSubtractionFedMode = cms.bool(False)
process.siStripZeroSuppression.Algorithms.CommonModeNoiseSubtractionMode = cms.string("IteratedMedian")
process.siStripZeroSuppression.doAPVRestore = cms.bool(True)
process.siStripZeroSuppression.produceRawDigis = cms.bool(True)
process.siStripZeroSuppression.produceCalculatedBaseline = cms.bool(True)
process.siStripZeroSuppression.storeCM = cms.bool(True)
process.siStripZeroSuppression.storeInZScollBadAPV = cms.bool(True)


    



process.TimerService = cms.Service("TimerService", useCPUtime = cms.untracked.bool(True) # set to false for wall-clock-time
)								  
								  
# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.siStripDigis)
process.reconstruction_step = cms.Path(process.striptrackerlocalreco)
process.endjob_step = cms.Path(process.endOfProcess)
process.RECOoutput_step = cms.EndPath(process.RECOoutput)
#process.Timer_step = cms.Path(process.myTimer)

# Schedule definition
#process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.hipmonitor_step, process.RECOoutput_step)
process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step, process.RECOoutput_step)
# customisation of the process


# Automatic addition of the customisation functionfrom SimGeneral.DataMixingModule.DataMixer_DataConditions_3_8_X_data2010
def customise(process):

#
# IOV set based on GlobalTag GR_R_35X_V8B
#
# placeholder !!!!!! replace with the actual run number of
# the real run to be overlaid

    process.source.firstRun = cms.untracked.uint32(1)

    process.stripConditions = cms.ESSource("PoolDBESSource",
        process.CondDBSetup,
        timetype = cms.untracked.string('runnumber'),
        connect = cms.string('frontier://FrontierProd/CMS_COND_31X_STRIP'),
        toGet = cms.VPSet(
          cms.PSet(
            record = cms.string('SiStripNoisesRcd'),
            tag = cms.string('SiStripNoise_GR10_v1_hlt')
          ),
          cms.PSet(
            record = cms.string('SiStripPedestalsRcd'),
            tag = cms.string('SiStripPedestals_GR10_v1_hlt')
          ),
          cms.PSet(
            record = cms.string('SiStripFedCablingRcd'),
            tag = cms.string('SiStripFedCabling_GR10_v1_hlt')
          ),
          cms.PSet(
            record = cms.string('SiStripBadChannelRcd'),
            tag = cms.string('SiStripBadChannel_FromOnline_GR10_v1_hlt')
          ),
          cms.PSet(
            record = cms.string('SiStripLatencyRcd'),
            tag = cms.string('SiStripLatency_GR10_v2_hlt')
          ),
          cms.PSet(
            record = cms.string('SiStripThresholdRcd'),
            tag = cms.string('SiStripThreshold_GR10_v1_hlt')
          ),
          cms.PSet(
            record = cms.string('SiStripBadFiberRcd'),
            tag = cms.string('SiStripBadChannel_FromOfflineCalibration_GR10_v2_offline')
          ),
          cms.PSet(
            record = cms.string('SiStripBadModuleRcd'),
            tag = cms.string('SiStripBadChannel_FromEfficiencyAnalysis_GR10_v1_offline')
          ),
          cms.PSet(
            record = cms.string('SiStripConfObjectRcd'),
            tag = cms.string('SiStripShiftAndCrosstalk_GR10_v1_offline')
          ),
          cms.PSet(
            record = cms.string('SiStripLorentzAngleRcd'),
            tag = cms.string('SiStripLorentzAngle_GR10_v1_offline')
          ),
          cms.PSet(
            record = cms.string('SiStripApvGain2Rcd'),
            tag = cms.string('SiStripApvGain_FromParticles_GR10_v2_offline')
          ),
          cms.PSet(
            record = cms.string('SiStripApvGainRcd'),
            tag = cms.string('SiStripApvGain_GR10_v1_hlt')
          )
        )
    )
    
    process.es_prefer_strips = cms.ESPrefer("PoolDBESSource","stripConditions")

    
    process.ecalConditions1 = cms.ESSource("PoolDBESSource",                                          
         process.CondDBSetup,                                                                         
         timetype = cms.string('runnumber'),                                                          
         toGet = cms.VPSet(                                                                           
             cms.PSet(                                                                                
        record = cms.string('EcalADCToGeVConstantRcd'),                                               
        tag = cms.string('EcalADCToGeVConstant_v6_offline')
        ),                                                                                            
             cms.PSet(                                                                                
        record = cms.string('EcalChannelStatusRcd'),                                                  
        tag = cms.string('EcalChannelStatus_v04_offline')                                   
        ),                                                                                            
             cms.PSet(                                                                                
        record = cms.string('EcalGainRatiosRcd'),                                                     
        tag = cms.string('EcalGainRatio_TestPulse2009_offline')                                      
        ),                                                                                            
             cms.PSet(                                                                                
        record = cms.string('EcalIntercalibConstantsRcd'),                                            
        tag = cms.string('EcalIntercalibConstants_v6_offline')                                 
        ),                                                                                            
             cms.PSet(                                                                                
        record = cms.string('EcalIntercalibErrorsRcd'),                                               
        tag = cms.string('EcalIntercalibErrors_mc')                                                   
        ),                                                                                            
             cms.PSet(                                                                                
        record = cms.string('EcalMappingElectronicsRcd'),                                             
        tag = cms.string('EcalMappingElectronics_EEMap')                                              
        ),                                                                                            
             cms.PSet(                                                                                
        record = cms.string('EcalPedestalsRcd'),                                                      
        tag = cms.string('EcalPedestals_2009runs_hlt')                                                
        ),                                                                                            
             cms.PSet(                                                                                
        record = cms.string('EcalTBWeightsRcd'),                                                      
        tag = cms.string('EcalTBWeights_EBEE_v01_offline')                                     
        ),                                                                                            
             cms.PSet(                                                                                
        record = cms.string('EcalTimeCalibConstantsRcd'),                                             
        tag = cms.string('EcalTimeCalibConstants_v02_offline')
        ),                                                                                            
             cms.PSet(                                                                                
        record = cms.string('EcalWeightXtalGroupsRcd'),                                               
        tag = cms.string('EcalWeightXtalGroups_EBEE_offline')                                   
        ),                                                                                            
             cms.PSet(                                                                   
        record = cms.string('EcalLaserAPDPNRatiosRcd'),                                               
        tag = cms.string('EcalLaserAPDPNRatios_p1p2p3_v2_mc')                                        
        ),                                                                                            
             ),                                                                                       
        connect = cms.string('frontier://FrontierProd/CMS_COND_31X_ECAL'),                            
              authenticationMethod = cms.untracked.uint32(0)                                          
    )                                                                                                 
                                                                                                      

    process.ecalConditions2 = cms.ESSource("PoolDBESSource",
                                           process.CondDBSetup,
                                           timetype = cms.string('runnumber'),
                                           toGet = cms.VPSet(
        cms.PSet(
        record = cms.string('EcalTPGCrystalStatusRcd'),
        tag = cms.string('EcalTPGCrystalStatus_v2_hlt')
        ),
        cms.PSet(
        record = cms.string('EcalTPGFineGrainEBGroupRcd'),
        tag = cms.string('EcalTPGFineGrainEBGroup_v2_hlt')
        ),
        cms.PSet(
        record = cms.string('EcalTPGFineGrainEBIdMapRcd'),
        tag = cms.string('EcalTPGFineGrainEBIdMap_v2_hlt')
        ),
        cms.PSet(
        record = cms.string('EcalTPGFineGrainStripEERcd'),
        tag = cms.string('EcalTPGFineGrainStripEE_v2_hlt')
        ),
        cms.PSet(
        record = cms.string('EcalTPGFineGrainTowerEERcd'),
        tag = cms.string('EcalTPGFineGrainTowerEE_v2_hlt')
        ),
        cms.PSet(
        record = cms.string('EcalTPGLinearizationConstRcd'),
        tag = cms.string('EcalTPGLinearizationConst_v2_hlt')
        ),
        cms.PSet(
        record = cms.string('EcalTPGLutGroupRcd'),
        tag = cms.string('EcalTPGLutGroup_v2_hlt')
        ),
        cms.PSet(
        record = cms.string('EcalTPGLutIdMapRcd'),
        tag = cms.string('EcalTPGLutIdMap_v2_hlt')
        ),
        cms.PSet(
        record = cms.string('EcalTPGPedestalsRcd'),
        tag = cms.string('EcalTPGPedestals_v2_hlt')
        ),
        cms.PSet(
        record = cms.string('EcalTPGPhysicsConstRcd'),
        tag = cms.string('EcalTPGPhysicsConst_v2_hlt')
        ),
        cms.PSet(
        record = cms.string('EcalTPGSlidingWindowRcd'),
        tag = cms.string('EcalTPGSlidingWindow_v2_hlt')
        ),
        cms.PSet(
        record = cms.string('EcalTPGTowerStatusRcd'),
        tag = cms.string('EcalTPGTowerStatus_hlt')
        ),
        cms.PSet(
        record = cms.string('EcalTPGWeightGroupRcd'),
        tag = cms.string('EcalTPGWeightGroup_v2_hlt')
        ),
        cms.PSet(
        record = cms.string('EcalTPGWeightIdMapRcd'),
        tag = cms.string('EcalTPGWeightIdMap_v2_hlt')
        ),
        ),
        connect = cms.string('frontier://FrontierProd/CMS_COND_34X_ECAL'),
               authenticationMethod = cms.untracked.uint32(0)
    )

    process.es_prefer_ecal1 = cms.ESPrefer("PoolDBESSource","ecalConditions1")                        
    process.es_prefer_ecal2 = cms.ESPrefer("PoolDBESSource","ecalConditions2")                        

                                                                                                      
    process.hcalConditions = cms.ESSource("PoolDBESSource",                                           
                                          process.CondDBSetup,                          
                                          timetype = cms.string('runnumber'),                         
                                          toGet = cms.VPSet(                                          
        cms.PSet(                                                                                     
        record = cms.string('HcalChannelQualityRcd'),                                                 
        tag = cms.string('HcalChannelQuality_v2.10_offline')                                          
        ),                                                                                            
        cms.PSet(                                                                                     
        record = cms.string('HcalElectronicsMapRcd'),                                                 
        tag = cms.string('HcalElectronicsMap_v7.03_hlt')                                              
        ),                                                                                            
        cms.PSet(                                                                                     
        record = cms.string('HcalGainsRcd'),                                                          
        tag = cms.string('HcalGains_v2.32_offline')                                                   
        ),                                                                                            
        cms.PSet(                                                                                     
        record = cms.string('HcalL1TriggerObjectsRcd'),                                               
        tag = cms.string('HcalL1TriggerObjects_v1.00_hlt')                                            
        ),                                                                                            
        cms.PSet(                                                                                     
        record = cms.string('HcalLUTCorrsRcd'),                                                       
        tag = cms.string('HcalLUTCorrs_v1.01_hlt')                                                    
        ),                                                                                            
        cms.PSet(                                                                                     
        record = cms.string('HcalPedestalsRcd'),                                                      
        tag = cms.string('HcalPedestals_ADC_v9.12_offline')                                        
        ),                                                                                            
        cms.PSet(                                                                                     
        record = cms.string('HcalPedestalWidthsRcd'),                                                 
        tag = cms.string('HcalPedestalWidths_ADC_v7.01_hlt')                                          
        ),                                                                                            
        cms.PSet(                                                                                     
        record = cms.string('HcalPFCorrsRcd'),                                                        
        tag = cms.string('HcalPFCorrs_v2.00_express')                                                 
        ),                                                                                            
        cms.PSet(                                                                                     
        record = cms.string('HcalQIEDataRcd'),                                                        
        tag = cms.string('HcalQIEData_NormalMode_v7.00_hlt')                                          
        ),                                                                                            
        cms.PSet(                                                                                     
        record = cms.string('HcalRespCorrsRcd'),                                                      
        tag = cms.string('HcalRespCorrs_v1.02_express')                                               
        ),                                                                                            
        cms.PSet(                                                                                     
        record = cms.string('HcalTimeCorrsRcd'),                                                      
        tag = cms.string('HcalTimeCorrs_v1.00_express')                                               
        ),                                                                                            
        cms.PSet(                                                                                     
        record = cms.string('HcalZSThresholdsRcd'),                                                   
        tag = cms.string('HcalZSThresholds_v1.01_hlt')                                                
        ),                                                                                            
        ),                                                                                            
             connect = cms.string('frontier://FrontierProd/CMS_COND_31X_HCAL'),                       
                      authenticationMethod = cms.untracked.uint32(0)                                  
    )                                                                                                 
                                                                                                      
    process.es_prefer_hcal = cms.ESPrefer("PoolDBESSource","hcalConditions")                          
                                                                                                      
    try: 
        process.ecalRecHit.ChannelStatusToBeExcluded = [ 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 78, 142 ]  
    except:
        return(process)
 
    return(process)


process = customise(process)


# End of customisation functions
