import FWCore.ParameterSet.Config as cms

# Entries for H2g skim
#
# Kati Lassila-Perini - Helsinki Institute of Physics
#
higgsTo2GammaFilter = cms.EDFilter("HiggsTo2GammaSkim",
    # Collection to be accessed
    PhotonCollectionLabel = cms.InputTag("correctedPhotons"),
    DebugHiggsTo2GammaSkim = cms.bool(False),
    # Minimum number of identified photons above pt threshold
    nPhotonMinimum = cms.int32(2),
    # Pt threshold for photons
    photon1MinimumPt = cms.double(15.0)
)


