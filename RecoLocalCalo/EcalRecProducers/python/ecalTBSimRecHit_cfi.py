import FWCore.ParameterSet.Config as cms

# rechit producer
ecalTBSimRecHit = cms.EDProducer("EcalRecHitProducer",
    EErechitCollection = cms.string('EcalRecHitsEE'),
    EEuncalibRecHitCollection = cms.InputTag("ecalTBSimWeightUncalibRecHit","EcalUncalibRecHitsEE"),
    EBuncalibRecHitCollection = cms.InputTag("ecalTBSimWeightUncalibRecHit","EcalUncalibRecHitsEB"),
    EBrechitCollection = cms.string('EcalRecHitsEB'),
    ChannelStatusToBeExcluded = cms.vint32(),
    algo = cms.string("EcalRecHitWorkerSimple"),
    killDeadChannels = cms.bool(True),
    laserCorrection = cms.bool(True),
    # reco flags association to DB flag
    # the vector index corresponds to the DB flag
    # the value correspond to the reco flag
    flagsMapDBReco = cms.vint32(
             0,   0,   0,  0, # standard reco
             4,               # faulty hardware (noisy)
            -1,  -1,  -1,     # not yet assigned
             4,   4,          # faulty hardware (fixed gain)
             6,   6,   6,     # dead channel with trigger
             7,               # dead FE
             8                # dead or recovery failed
            ),
    # for channel recovery
    algoRecover = cms.string("EcalRecHitWorkerRecover"),
    recoverEBIsolatedChannels = cms.bool(False),
    recoverEEIsolatedChannels = cms.bool(False),
    recoverEBVFE  = cms.bool(False),
    recoverEEVFE  = cms.bool(False),
    recoverEBFE = cms.bool(False),
    recoverEEFE = cms.bool(False),
    ebDetIdToBeRecovered = cms.InputTag("ebDetId"),
    eeDetIdToBeRecovered = cms.InputTag("eeDetId"),
    ebFEToBeRecovered = cms.InputTag("ebFE"),
    eeFEToBeRecovered = cms.InputTag("eeFE"),
    singleChannelRecoveryMethod = cms.string("NeuralNetworks"),
    singleChannelRecoveryThreshold = cms.double(0),
    triggerPrimitiveDigiCollection = cms.InputTag("ecalDigis")
)
