import FWCore.ParameterSet.Config as cms

# rechit producer
ecalRecHit = cms.EDProducer("EcalRecHitProducer",
    EErechitCollection = cms.string('EcalRecHitsEE'),
    EEuncalibRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE"),
    EBuncalibRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB"),
    EBrechitCollection = cms.string('EcalRecHitsEB'),
    # channel flags to be exluded from reconstruction, e.g { 1, 2 }
    ChannelStatusToBeExcluded = cms.vint32(),
    algo = cms.string("EcalRecHitWorkerSimple"),
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
    recoverEBIsolatedChannels = cms.bool(True),
    recoverEEIsolatedChannels = cms.bool(True),
    recoverEBFE = cms.bool(True),
    recoverEEFE = cms.bool(True),
    ebDetIdToBeRecovered = cms.InputTag("ecalDetIdToBeRecovered:ebDetId"),
    eeDetIdToBeRecovered = cms.InputTag("ecalDetIdToBeRecovered:eeDetId"),
    ebFEToBeRecovered = cms.InputTag("ecalDetIdToBeRecovered:ebFE"),
    eeFEToBeRecovered = cms.InputTag("ecalDetIdToBeRecovered:eeFE"),
    singleChannelRecoveryMethod = cms.string("NeuralNetworks"),
    singleChannelRecoveryThreshold = cms.double(0),
    triggerPrimitiveDigiCollection = cms.InputTag("ecalDigis")
)
