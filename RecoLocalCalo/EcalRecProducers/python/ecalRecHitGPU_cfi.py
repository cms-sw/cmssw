import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.EcalRecAlgos.ecalCleaningAlgo import cleaningAlgoConfig 

# rechit producer
ecalRecHitGPU = cms.EDProducer("EcalRecHitProducerGPU",
                               
    uncalibrecHitsInLabelEB = cms.InputTag("ecalUncalibRecHitProducerGPU","EcalUncalibRecHitsEB"),
    uncalibrecHitsInLabelEE = cms.InputTag("ecalUncalibRecHitProducerGPU","EcalUncalibRecHitsEE"),
          
    #recHitsLabelEB = cms.string("EcalRecHitsGPUEB"),
    #recHitsLabelEE = cms.string("EcalRecHitsGPUEE"),
    recHitsLabelEB = cms.string("EcalRecHitsEB"),
    recHitsLabelEE = cms.string("EcalRecHitsEE"),
 
    maxNumberHits = cms.uint32(20000),  # FIXME AM
  
  
    #EErechitCollection = cms.string('EcalRecHitsEE'),
    #EEuncalibRecHitCollection = cms.InputTag("ecalMultiFitUncalibRecHit","EcalUncalibRecHitsEE"),
    #EBuncalibRecHitCollection = cms.InputTag("ecalMultiFitUncalibRecHit","EcalUncalibRecHitsEB"),
    #EBrechitCollection = cms.string('EcalRecHitsEB'),
   
    ## db statuses to be exluded from reconstruction (some will be recovered)
    ChannelStatusToBeExcluded = cms.vstring(   'kDAC',
                                               'kNoisy',
                                               'kNNoisy',
                                               'kFixedG6',
                                               'kFixedG1',
                                               'kFixedG0',
                                               'kNonRespondingIsolated',
                                               'kDeadVFE',
                                               'kDeadFE',
                                               'kNoDataNoTP',
                                               #
                                               # AM should I add them here?????
                                               # next ones from "flagsMapDBReco"
                                               # but not defined in "EcalChannelStatusCode.h"
                                               # but they are defined in "EcalRecHit.h"
                                               #
                                               #'kKilled',
                                               #'kTPSaturated',
                                               #'kL1SpikeFlag',
                                               ),
    
    ## avoid propagation of dead channels other than after recovery
    killDeadChannels = cms.bool(True),
    #algo = cms.string("EcalRecHitWorkerSimple"),
    
    ## define maximal and minimal values for the laser corrections
    
    EBLaserMIN = cms.double(0.01),                    #    EBLaserMIN = cms.double(0.5),
    EELaserMIN = cms.double(0.01),                    #    EELaserMIN = cms.double(0.5),
                                                     
    EBLaserMAX = cms.double(30.0),                    #    EBLaserMAX = cms.double(3.0),
    EELaserMAX = cms.double(30.0),                    #    EELaserMAX = cms.double(8.0),


    ## useful if time is not calculated, as at HLT                        
    #skipTimeCalib = cms.bool(False),                         

    ## apply laser corrections
    #laserCorrection = cms.bool(True),
                            
    ## reco flags association to DB flag
    flagsMapDBReco = cms.PSet(
        kGood  = cms.vstring('kOk','kDAC','kNoLaser','kNoisy'),
        kNoisy = cms.vstring('kNNoisy','kFixedG6','kFixedG1'),
        kNeighboursRecovered = cms.vstring('kFixedG0',
                                           'kNonRespondingIsolated',
                                           'kDeadVFE'),
        kTowerRecovered = cms.vstring('kDeadFE'),
        kDead           = cms.vstring('kNoDataNoTP')
        ), 
        
#//         flagmask_ |= 0x1 << EcalRecHit::kNeighboursRecovered;
#//         flagmask_ |= 0x1 << EcalRecHit::kTowerRecovered;
#//         flagmask_ |= 0x1 << EcalRecHit::kDead;
#//         flagmask_ |= 0x1 << EcalRecHit::kKilled;
#//         flagmask_ |= 0x1 << EcalRecHit::kTPSaturated;
#//         flagmask_ |= 0x1 << EcalRecHit::kL1SpikeFlag;


                            
    ## for channel recovery
    #algoRecover = cms.string("EcalRecHitWorkerRecover"),
    recoverEBIsolatedChannels = cms.bool(False),
    recoverEEIsolatedChannels = cms.bool(False),
    recoverEBVFE  = cms.bool(False),
    recoverEEVFE  = cms.bool(False),
    recoverEBFE = cms.bool(True),
    recoverEEFE = cms.bool(True),

    ##db statuses for which recovery in EE/EB should not be attempted           
    #dbStatusToBeExcludedEE = cms.vint32(
                                        #14,  # dead, no TP
                                        #78,  # dead, HV off
                                        #142, # dead,LV off
                                        #), 
    #dbStatusToBeExcludedEB = cms.vint32(
                                        #14,  # dead, no TP
                                        #78,  # dead, HV off
                                        #142, # dead,LV off
                                        #), 
    
    ## --- logWarnings for saturated DeadFEs
    ## if the logWarningThreshold is negative the Algo will not try recovery (in EE is not tested we may need negative threshold e.g. -1.e+9)
    ## if you want to enable recovery but you don't wish to throw logWarnings put the logWarningThresholds very high e.g +1.e+9
    ##  ~64 GeV is the TP saturation level
    #logWarningEtThreshold_EB_FE = cms.double(50),# in EB logWarningThreshold is actually in E (GeV)
    #logWarningEtThreshold_EE_FE = cms.double(50),# in EE the energy should correspond to Et (GeV) but the recovered values of energies are not tested if make sense
    #ebDetIdToBeRecovered = cms.InputTag("ecalDetIdToBeRecovered:ebDetId"),
    #eeDetIdToBeRecovered = cms.InputTag("ecalDetIdToBeRecovered:eeDetId"),
    #ebFEToBeRecovered = cms.InputTag("ecalDetIdToBeRecovered:ebFE"),
    #eeFEToBeRecovered = cms.InputTag("ecalDetIdToBeRecovered:eeFE"),
    #singleChannelRecoveryMethod = cms.string("NeuralNetworks"),
    #singleChannelRecoveryThreshold = cms.double(8),
    #triggerPrimitiveDigiCollection = cms.InputTag("ecalDigis:EcalTriggerPrimitives"),
    #cleaningConfig=cleaningAlgoConfig,

    )



#from Configuration.Eras.Modifier_fastSim_cff import fastSim
## no flags for bad channels in FastSim
#fastSim.toModify(ecalRecHit, 
                 #killDeadChannels = False,
                 #recoverEBFE = False,
                 #recoverEEFE = False)


