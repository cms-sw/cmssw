import FWCore.ParameterSet.Config as cms

# Entries for RS2g skim
rsTo2GammaFilter = cms.EDFilter("HiggsTo2GammaSkim",
    # Collection to be accessed
    PhotonCollectionLabel = cms.InputTag("correctedPhotons"),
    DebugHiggsTo2GammaSkim = cms.bool(False),
    # Minimum number of identified photons above pt threshold
    nPhotonMinimum = cms.int32(1),
    # Pt threshold for photons
    photon1MinimumPt = cms.double(50.0)
)


