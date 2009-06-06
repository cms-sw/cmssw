import FWCore.ParameterSet.Config as cms

# Entries for H2g skim
#
# Kati Lassila-Perini - Helsinki Institute of Physics
#
higgsTo2GammaFilter = cms.EDFilter("HiggsTo2GammaSkim",
    # Collection to be accessed
    PhotonCollectionLabel = cms.InputTag("photons"),
    DebugHiggsTo2GammaSkim = cms.bool(False),
    # Minimum number of identified photons above pt threshold
    #nPhotonMinimum = cms.int32(2),
    # Pt threshold for photons
    #photon1MinimumPt = cms.double(10.0)

   nPhotonLooseMin = cms.int32(2),
   nPhotonTightMin = cms.int32(0),
   photonLooseMinPt = cms.double(10.0),
   photonTightMinPt = cms.double(15.0),
   photonLooseMaxEta = cms.double(3.1),
   photonTightMaxEta = cms.double(2.6),
   photonLooseMaxHoE = cms.double(-1.0),
   photonTightMaxHoE = cms.double(0.2),
   photonLooseMaxHIsol = cms.double(-1.0),
   photonTightMaxHIsol = cms.double(15.0),
   photonLooseMaxEIsol = cms.double(-1.0),
   photonTightMaxEIsol = cms.double(10.0),
   photonLooseMaxTIsol = cms.double(-1.0),
   photonTightMaxTIsol = cms.double(5.0)
       
)


