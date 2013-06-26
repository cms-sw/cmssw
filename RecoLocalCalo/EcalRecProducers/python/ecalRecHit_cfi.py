import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.EcalRecAlgos.ecalCleaningAlgo import cleaningAlgoConfig 

# rechit producer
ecalRecHit = cms.EDProducer("EcalRecHitProducer",
    EErechitCollection = cms.string('EcalRecHitsEE'),
    EEuncalibRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE"),
    EBuncalibRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB"),
    EBrechitCollection = cms.string('EcalRecHitsEB'),
    # channel flags to be exluded from reconstruction, e.g { 1, 2 }
    ChannelStatusToBeExcluded = cms.vint32(),
    # avoid propagation of dead channels other than after recovery
    killDeadChannels = cms.bool(True),
    algo = cms.string("EcalRecHitWorkerSimple"),
    # define maximal and minimal values for the laser corrections
    
    EBLaserMIN = cms.double(0.5),
    EELaserMIN = cms.double(0.5),

    EBLaserMAX = cms.double(3.0),
    EELaserMAX = cms.double(8.0),


    # apply laser corrections
    laserCorrection = cms.bool(True),
    # reco flags association to DB flag
    # the vector index corresponds to the DB flag
    # the value correspond to the reco flag
    flagsMapDBReco = cms.vint32(
             0,   0,   0,  0, # standard reco
             4,               # faulty hardware (noisy)
            -1,  -1,  -1,     # not yet assigned
             4,   4,          # faulty hardware (fixed gain)
             7,   7,   7,     # dead channel with trigger
             8,               # dead FE
             9                # dead or recovery failed
            ),                        
                            
    # for channel recovery
    algoRecover = cms.string("EcalRecHitWorkerRecover"),
    recoverEBIsolatedChannels = cms.bool(False),
    recoverEEIsolatedChannels = cms.bool(False),
    recoverEBVFE  = cms.bool(False),
    recoverEEVFE  = cms.bool(False),
    recoverEBFE = cms.bool(True),
    recoverEEFE = cms.bool(True),
    #db statuses for which recovery in EE/EB should not be attempted           
    dbStatusToBeExcludedEE = cms.vint32(
                                        14,  # dead, no TP
                                        78,  # dead, HV off
                                        142, # dead,LV off
                                        ), 
    dbStatusToBeExcludedEB = cms.vint32(
                                        14,  # dead, no TP
                                        78,  # dead, HV off
                                        142, # dead,LV off
                                        ), 
    # --- logWarnings for saturated DeadFEs
    # if the logWarningThreshold is negative the Algo will not try recovery (in EE is not tested we may need negative threshold e.g. -1.e+9)
    # if you want to enable recovery but you don't wish to throw logWarnings put the logWarningThresholds very high e.g +1.e+9
    #  ~64 GeV is the TP saturation level
    logWarningEtThreshold_EB_FE = cms.double(50),# in EB logWarningThreshold is actually in E (GeV)
    logWarningEtThreshold_EE_FE = cms.double(50),# in EE the energy should correspond to Et (GeV) but the recovered values of energies are not tested if make sense
    ebDetIdToBeRecovered = cms.InputTag("ecalDetIdToBeRecovered:ebDetId"),
    eeDetIdToBeRecovered = cms.InputTag("ecalDetIdToBeRecovered:eeDetId"),
    ebFEToBeRecovered = cms.InputTag("ecalDetIdToBeRecovered:ebFE"),
    eeFEToBeRecovered = cms.InputTag("ecalDetIdToBeRecovered:eeFE"),
    singleChannelRecoveryMethod = cms.string("NeuralNetworks"),
    singleChannelRecoveryThreshold = cms.double(8),
    triggerPrimitiveDigiCollection = cms.InputTag("ecalDigis:EcalTriggerPrimitives"),
    cleaningConfig=cleaningAlgoConfig,

    )
