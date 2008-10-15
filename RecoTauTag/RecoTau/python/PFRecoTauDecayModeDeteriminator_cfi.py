import FWCore.ParameterSet.Config as cms

pfTauDecayModeInsideOut = cms.EDProducer("PFRecoTauDecayModeDeterminator",
    maxPiZeroMass = cms.double(0.2),
    refitTracks = cms.bool(False),
    minPtFractionForThirdGamma = cms.double(-1.0),
    mergeLowPtPhotonsFirst = cms.bool(True),
    filterPhotons = cms.bool(False),
    PFTauProducer = cms.InputTag("pfRecoTauProducerInsideOut"),
    maxPhotonsToMerge = cms.uint32(3),
    filterTwoProngs = cms.bool(False),
    minPtFractionForSecondProng = cms.double(-1.0),
    maxDistance = cms.double(0.01),
    maxNbrOfIterations = cms.int32(10)
)

pfTauDecayModeHighEfficiency = cms.EDProducer("PFRecoTauDecayModeDeterminator",
    maxPiZeroMass = cms.double(0.2),
    refitTracks = cms.bool(False),
    minPtFractionForThirdGamma = cms.double(-1.0),
    mergeLowPtPhotonsFirst = cms.bool(True),
    filterPhotons = cms.bool(False),
    PFTauProducer = cms.InputTag("pfRecoTauProducerHighEfficiency"),
    maxPhotonsToMerge = cms.uint32(3),
    filterTwoProngs = cms.bool(False),
    minPtFractionForSecondProng = cms.double(-1.0),
    maxDistance = cms.double(0.01),
    maxNbrOfIterations = cms.int32(10)
)


