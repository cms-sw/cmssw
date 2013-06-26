import FWCore.ParameterSet.Config as cms

ecalMipGraphs = cms.EDAnalyzer("EcalMipGraphs",
    # parameter for the amplitude threshold
    amplitudeThreshold = cms.untracked.double(0.5),
    EBDigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
    EEDigiCollection = cms.InputTag("ecalEBunpacker","eeDigis"),
    EcalRecHitCollectionEB = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    EcalRecHitCollectionEE = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    headerProducer = cms.InputTag("ecalEBunpacker"),
    #EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEB"),
    # masked EBids
    maskedEBs = cms.untracked.vstring('-1'),
    # masked FEDs
    maskedFEDs = cms.untracked.vint32(-1),
    # use hash index to mask channels
    # add a simple description of hashIndex (hhahhahhh...)
    maskedChannels = cms.untracked.vint32(),
    # parameter for fixed crystals mode (use hashedIndices)
    seedCrys = cms.untracked.vint32(),
    # parameter for size of the square matrix, i.e.,
    # should the seed be at the center of a 3x3 matrix, a 5x5, etc.
    # must be an odd number (default is 3)
    side = cms.untracked.int32(3),
    minimumTimingAmplitude = cms.untracked.double(0.100)
)


