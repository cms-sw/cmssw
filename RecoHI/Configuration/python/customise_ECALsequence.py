import FWCore.ParameterSet.Config as cms

def changeHeavyIonsToUseECALGlobalFit(process) :
    if hasattr (process, "caloReco") :
        process.load('RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi')
        process.ecalUncalibRecHitSequenceHI = cms.Sequence(process.ecalGlobalUncalibRecHit*
                                                           process.ecalDetIdToBeRecovered)
        process.ecalLocalRecoSequenceHI     = cms.Sequence(process.ecalUncalibRecHitSequenceHI*
                                                           process.ecalRecHitSequence)
        process.ecalRecHit.EEuncalibRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE")
        process.ecalRecHit.EBuncalibRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB")
        process.caloReco.replace(process.ecalUncalibRecHitSequence, process.ecalUncalibRecHitSequenceHI)

    if hasattr (process, "ecalMonitorTask") :
        process.ecalMonitorTask.collectionTags.EBUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB")
        process.ecalMonitorTask.collectionTags.EEUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE")
        process.ecalMonitorTask.collectionTags.EBLaserLedUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB")
        process.ecalMonitorTask.collectionTags.EELaserLedUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE")

    return process
