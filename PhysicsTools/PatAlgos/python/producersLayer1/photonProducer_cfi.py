import FWCore.ParameterSet.Config as cms

allLayer1Photons = cms.EDProducer("PATPhotonProducer",
    # General configurables
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
      userFunctions = cms.vstring(""),
      userFunctionLabels = cms.vstring("")
    ),

    embedSuperCluster = cms.bool(True), ## whether to embed in AOD externally stored supercluster

    # Isolation configurables
    #   store isolation values
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
    #   store isodeposits to recompute isolation
    isoDeposits = cms.PSet(
        tracker = cms.InputTag("gamIsoDepositTk"),
        ecal    = cms.InputTag("gamIsoDepositEcalFromHits"),
        hcal    = cms.InputTag("gamIsoDepositHcalFromTowers"),
    ),

    # Photon ID configurables
    addPhotonID = cms.bool(True),
    photonIDSource = cms.InputTag("PhotonIDProd","PhotonAssociatedID"), ## ValueMap<reco::PhotonID> keyed to photonSource

    # Trigger matching configurables
    addTrigMatch = cms.bool(True),
    trigPrimMatch = cms.VInputTag(cms.InputTag("photonTrigMatchHLT1PhotonRelaxed")),

    # MC matching configurables
    addGenMatch = cms.bool(True),
    embedGenMatch = cms.bool(False),
    genParticleMatch = cms.InputTag("photonMatch"), ## particles source to be used for the matching

    # Efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),

    # Resolutions
    addResolutions  = cms.bool(False),
)


