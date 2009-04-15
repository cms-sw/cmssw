import FWCore.ParameterSet.Config as cms

allLayer1Photons = cms.EDProducer("PATPhotonProducer",
    # input collection
    photonSource = cms.InputTag("photons"),
                                 
    # user data to add
    userData = cms.PSet(
      # add custom classes here
      userClasses = cms.PSet(
        src = cms.VInputTag('')
      ),
      # add doubles here
      userFloats = cms.PSet(
        src = cms.VInputTag('')
      ),
      # add ints here
      userInts = cms.PSet(
        src = cms.VInputTag('')
      ),
      # add "inline" functions here
      userFunctions = cms.vstring(),
      userFunctionLabels = cms.vstring()
    ),

    # embedding of AOD items
    embedSuperCluster = cms.bool(True), ## whether to embed in AOD externally stored supercluster

    # isolation
    isolation = cms.PSet(
        tracker = cms.PSet(
            src = cms.InputTag("gamIsoFromDepsTk"),
        ),
        ecal = cms.PSet(
            src = cms.InputTag("gamIsoFromDepsEcalFromHits"),
        ),
        hcal = cms.PSet(
            src = cms.InputTag("gamIsoFromDepsHcalFromTowers"),
        ),
        user = cms.VPSet(),
    ),
    # embed IsoDeposits to recompute isolation
    isoDeposits = cms.PSet(
        tracker = cms.InputTag("gamIsoDepositTk"),
        ecal    = cms.InputTag("gamIsoDepositEcalFromHits"),
        hcal    = cms.InputTag("gamIsoDepositHcalFromTowers"),
    ),

    # photon ID
    addPhotonID = cms.bool(False),
    photonIDSource = cms.InputTag("PhotonIDProd","PhotonAssociatedID"), ## ValueMap<reco::PhotonID> keyed to photonSource

    # trigger matching
    addTrigMatch = cms.bool(False),
    trigPrimMatch = cms.VInputTag(''),

    # mc matching
    addGenMatch = cms.bool(True),
    embedGenMatch = cms.bool(False),
    genParticleMatch = cms.InputTag("photonMatch"), ## particles source to be used for the matching

    # efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),

    # resolutions
    addResolutions  = cms.bool(False),
)


