import FWCore.ParameterSet.Config as cms

patPhotons = cms.EDProducer("PATPhotonProducer",
    # input collection
    photonSource = cms.InputTag("gedPhotons"),
                                 
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
      # add candidate ptrs here
      userCands = cms.PSet(
        src = cms.VInputTag('')
      ),
      # add "inline" functions here
      userFunctions = cms.vstring(),
      userFunctionLabels = cms.vstring()
    ),

    # embedding of AOD items
    embedSuperCluster = cms.bool(True), ## whether to embed in AOD externally stored supercluster

    # embed IsoDeposits to recompute isolation
    isoDeposits = cms.PSet(),

    # user defined isolation variables the variables defined here will be accessible
    # via pat::Photon::userIsolation(IsolationKeys key) with the key as defined in
    # DataFormats/PatCandidates/interface/Isolation.h
    userIsolation = cms.PSet(),

    # photon ID
    addPhotonID = cms.bool(True),
    photonIDSources = cms.PSet(
             PhotonCutBasedIDLoose = cms.InputTag('PhotonIDProdGED',
                                                  'PhotonCutBasedIDLoose'),
             PhotonCutBasedIDTight = cms.InputTag('PhotonIDProdGED',
                                                  'PhotonCutBasedIDTight')
           ),
    # mc matching
    addGenMatch = cms.bool(True),
    embedGenMatch = cms.bool(True),
    genParticleMatch = cms.InputTag("photonMatch"), ## particles source to be used for the matching

    # efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),

    # resolutions
    addResolutions  = cms.bool(False),
    resolutions     = cms.PSet()

)
