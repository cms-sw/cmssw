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

hfnegTowers = cms.Sequence(
	towersAboveThreshold *
	hfNegTowers)	

hfposTowers = cms.Sequence(
	towersAboveThreshold *
	hfPosTowers)	


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

hfposFilter = cms.Sequence(
	hfposTowers +
	hfPosFilter)

hfnegFilter = cms.Sequence(
	hfnegTowers +
	hfNegFilter)


# one HF tower above threshold on each side
hfCoincFilter = cms.Sequence(
    towersAboveThreshold *
    hfPosTowers *
    hfNegTowers *
    hfPosFilter *
    hfNegFilter)


# two HF towers above threshold on each side
hfPosFilter2 = hfPosFilter.clone(minNumber=cms.uint32(2))
hfNegFilter2 = hfNegFilter.clone(minNumber=cms.uint32(2))

hfposFilter2 = cms.Sequence(
	hfposTowers +
	hfPosFilter2)

hfnegFilter2 = cms.Sequence(
	hfnegTowers +
	hfNegFilter2)

hfCoincFilter2 = cms.Sequence(
    towersAboveThreshold *
    hfPosTowers *
    hfNegTowers *
    hfPosFilter2 *
    hfNegFilter2)

#three HF towers above threshold on each side
hfPosFilter3 = hfPosFilter.clone(minNumber=cms.uint32(3))
hfNegFilter3 = hfNegFilter.clone(minNumber=cms.uint32(3))

hfposFilter3 = cms.Sequence(
	hfposTowers +
	hfPosFilter3)

hfnegFilter3 = cms.Sequence(
	hfnegTowers +
	hfNegFilter3)


hfCoincFilter3 = cms.Sequence(
    towersAboveThreshold *
    hfPosTowers *
    hfNegTowers *
    hfPosFilter3 *
    hfNegFilter3)

#four HF towers above threshold on each side
hfPosFilter4 = hfPosFilter.clone(minNumber=cms.uint32(4))
hfNegFilter4 = hfNegFilter.clone(minNumber=cms.uint32(4))

hfposFilter4 = cms.Sequence(
	hfposTowers +
	hfPosFilter4)

hfnegFilter4 = cms.Sequence(
	hfnegTowers +
	hfNegFilter4)

hfCoincFilter4 = cms.Sequence(
    towersAboveThreshold *
    hfPosTowers *
    hfNegTowers *
    hfPosFilter4 *
    hfNegFilter4)

#five hf towers above threshold on each side
hfPosFilter5 = hfPosFilter.clone(minNumber=cms.uint32(5))
hfNegFilter5 = hfNegFilter.clone(minNumber=cms.uint32(5))

hfposFilter5 = cms.Sequence(
	hfposTowers +
	hfPosFilter5)

hfnegFilter5 = cms.Sequence(
	hfnegTowers +
	hfNegFilter5)

hfCoincFilter5 = cms.Sequence(
    towersAboveThreshold *
    hfPosTowers *
    hfNegTowers *
    hfPosFilter5 *
    hfNegFilter5)
