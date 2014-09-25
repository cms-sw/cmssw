import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.EcalRecAlgos.ecalCleaningAlgo import cleaningAlgoConfig 

# rechit producer
ecalRecHit = cms.EDProducer("EcalRecHitProducer",
    EErechitCollection = cms.string('EcalRecHitsEE'),
    EEuncalibRecHitCollection = cms.InputTag("ecalMultiFitUncalibRecHit","EcalUncalibRecHitsEE"),
    EBuncalibRecHitCollection = cms.InputTag("ecalMultiFitUncalibRecHit","EcalUncalibRecHitsEB"),
    EBrechitCollection = cms.string('EcalRecHitsEB'),
    # db statuses to be exluded from reconstruction (some will be recovered)
    ChannelStatusToBeExcluded = cms.vstring(   'kNoisy',
                                               'kNNoisy',
                                               'kFixedG6',
                                               'kFixedG1',
                                               'kFixedG0',
                                               'kNonRespondingIsolated',
                                               'kDeadVFE',
                                               'kDeadFE',
                                               'kNoDataNoTP',),
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
    flagsMapDBReco = cms.PSet(
        kGood  = cms.vstring('kOk','kDAC','kNoLaser','kNoisy'),
        kNoisy = cms.vstring('kNNoisy','kFixedG6','kFixedG1'),
        kNeighboursRecovered = cms.vstring('kFixedG0',
										   'kNonRespondingIsolated',
										   'kDeadVFE'),
        kTowerRecovered = cms.vstring('kDeadFE'),
        kDead           = cms.vstring('kNoDataNoTP')
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
