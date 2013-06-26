import FWCore.ParameterSet.Config as cms

# make calotowers into candidates
towersAboveThreshold = cms.EDProducer("CaloTowerCandidateCreator",
    src = cms.InputTag("towerMaker"),
    verbose = cms.untracked.int32(0),
    minimumE = cms.double(3.0),
    minimumEt = cms.double(0.0),
)

# select HF+ towers above threshold
hfPosTowers = cms.EDFilter("EtaPtMinCandSelector",
    src = cms.InputTag("towersAboveThreshold"),
    ptMin   = cms.double(0),
    etaMin = cms.double(3.0),
    etaMax = cms.double(6.0)
)

# select HF- towers above threshold
hfNegTowers = cms.EDFilter("EtaPtMinCandSelector",
    src = cms.InputTag("towersAboveThreshold"),
    ptMin   = cms.double(0),
    etaMin = cms.double(-6.0),
    etaMax = cms.double(-3.0)
)

# require at least one HF+ tower above threshold
hfPosFilter = cms.EDFilter("CandCountFilter",
    src = cms.InputTag("hfPosTowers"),
    minNumber = cms.uint32(1)
)

# require at least one HF- tower above threshold
hfNegFilter = cms.EDFilter("CandCountFilter",
    src = cms.InputTag("hfNegTowers"),
    minNumber = cms.uint32(1)
)

# one HF tower above threshold on each side
hfCoincFilter = cms.Sequence(
    towersAboveThreshold *
    hfPosTowers *
    hfNegTowers *
    hfPosFilter *
    hfNegFilter)


# three HF towers above threshold on each side

hfPosFilter3 = hfPosFilter.clone(minNumber=cms.uint32(3))
hfNegFilter3 = hfNegFilter.clone(minNumber=cms.uint32(3))

hfCoincFilter3 = cms.Sequence(
    towersAboveThreshold *
    hfPosTowers *
    hfNegTowers *
    hfPosFilter3 *
    hfNegFilter3)
