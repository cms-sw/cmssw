import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.EcalRecAlgos.ecalCleaningAlgo import cleaningAlgoConfig 

# rechit producer
ecalRecHitGPU = cms.EDProducer("EcalRecHitProducerGPU",
                               
    uncalibrecHitsInLabelEB = cms.InputTag("ecalUncalibRecHitProducerGPU","EcalUncalibRecHitsEB"),
    uncalibrecHitsInLabelEE = cms.InputTag("ecalUncalibRecHitProducerGPU","EcalUncalibRecHitsEE"),
          
    recHitsLabelEB = cms.string("EcalRecHitsEB"),
    recHitsLabelEE = cms.string("EcalRecHitsEE"),
 
    maxNumberHitsEB = cms.uint32(61200),
    maxNumberHitsEE = cms.uint32(14648),  
  
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
    
    ## define maximal and minimal values for the laser corrections
    
    EBLaserMIN = cms.double(0.01),
    EELaserMIN = cms.double(0.01),

    EBLaserMAX = cms.double(30.0),
    EELaserMAX = cms.double(30.0),

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

    ## for channel recovery
    recoverEBIsolatedChannels = cms.bool(False),
    recoverEEIsolatedChannels = cms.bool(False),
    recoverEBVFE  = cms.bool(False),
    recoverEEVFE  = cms.bool(False),
    recoverEBFE = cms.bool(True),
    recoverEEFE = cms.bool(True),
)

