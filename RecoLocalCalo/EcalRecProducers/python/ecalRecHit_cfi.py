import FWCore.ParameterSet.Config as cms

# rechit producer
ecalRecHit = cms.EDProducer("EcalRecHitProducer",
    EErechitCollection = cms.string('EcalRecHitsEE'),
    EEuncalibRecHitCollection = cms.InputTag("ecalUncalibRecHit","EcalUncalibRecHitsEE"),
    EBuncalibRecHitCollection = cms.InputTag("ecalUncalibRecHit","EcalUncalibRecHitsEB"),
    EBrechitCollection = cms.string('EcalRecHitsEB'),
    # channel flags to be exluded from reconstruction, e.g { 1, 2 }
    ChannelStatusToBeExcluded = cms.vint32(),
    algo = cms.string("EcalRecHitWorkerSimple")
)

#ecalRecHit = cms.EDProducer("EcalRecHitProducer2",
#                            algo = cms.string('EcalRecHitSimpleAlgoWrapper'),
#                            EErechitCollection = cms.string('EcalRecHitsEE'),
#                            EEuncalibRecHitCollection = cms.InputTag("ecalWeightUncalibRecHit","EcalUncalibRecHitsEE"),
#                            EBuncalibRecHitCollection = cms.InputTag("ecalWeightUncalibRecHit","EcalUncalibRecHitsEB"),
#                            EBrechitCollection = cms.string('EcalRecHitsEB'),
#                            # channel flags to be exluded from reconstruction, e.g { 1, 2 }
#                            ChannelStatusToBeExcluded = cms.vint32()
#                            )
                         

