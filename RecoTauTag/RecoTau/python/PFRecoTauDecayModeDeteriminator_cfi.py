import FWCore.ParameterSet.Config as cms
import copy

pfTauDecayModeInsideOut = cms.EDProducer("PFRecoTauDecayModeDeterminator",
    maxPiZeroMass = cms.double(0.2),
    refitTracks = cms.bool(False),
    minPtFractionForThirdGamma = cms.double(-1.0),
    mergeLowPtPhotonsFirst = cms.bool(True),                    # as opposed to highest pt first
    filterPhotons = cms.bool(False),                            # not yet implemented
    PFTauProducer = cms.InputTag("pfRecoTauProducerInsideOut"),
    maxPhotonsToMerge = cms.uint32(3),
    filterTwoProngs = cms.bool(True),
    minPtFractionForSecondProng = cms.double(0.1),              # second prong pt/lead track pt fraction
    maxDistance = cms.double(0.01),
    maxNbrOfIterations = cms.int32(10)
)

pfTauDecayModeInsideOutTiny = copy.deepcopy(pfTauDecayModeInsideOut)
pfTauDecayModeInsideOutTiny.PFTauProducer = cms.InputTag("pfRecoTauProducerInsideOutTiny")

pfTauDecayModeHighEfficiency = cms.EDProducer("PFRecoTauDecayModeDeterminator",
    maxPiZeroMass = cms.double(0.2),
    refitTracks = cms.bool(False),
    minPtFractionForThirdGamma = cms.double(-1.0),
    mergeLowPtPhotonsFirst = cms.bool(True),
    filterPhotons = cms.bool(False),
    PFTauProducer = cms.InputTag("pfRecoTauProducerHighEfficiency"),
    maxPhotonsToMerge = cms.uint32(3),
    filterTwoProngs = cms.bool(True),
    minPtFractionForSecondProng = cms.double(-1.0),#second prong pt/lead track pt fraction
    maxDistance = cms.double(0.01),
    maxNbrOfIterations = cms.int32(10)
)


