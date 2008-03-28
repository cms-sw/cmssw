# The following comments couldn't be translated into the new config version:

# masked EBids

import FWCore.ParameterSet.Config as cms

ecalMipGraphs = cms.EDFilter("EcalMipGraphs",
    # parameter for the amplitude threshold
    amplitudeThreshold = cms.untracked.double(12.0),
    EBDigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
    # parameter for the name of the output root file with TH1F
    fileName = cms.untracked.string('ecalMipGraphs-'),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEB"),
    maskedEBs = cms.untracked.vstring('-1'),
    # masked FEDs
    maskedFEDs = cms.untracked.vint32(-1),
    # use hash index to mask channels
    # add a simple description of hashIndex (hhahhahhh...)
    maskedChannels = cms.untracked.vint32(),
    # parameter for fixed crystal mode (use hashedIndex)
    seedCry = cms.untracked.int32(0),
    # parameter for size of the square matrix, i.e.,
    # should the seed be at the center of a 3x3 matrix, a 5x5, etc.
    # must be an odd number (default is 3)
    side = cms.untracked.int32(3)
)


