import FWCore.ParameterSet.Config as cms

patPhotons = cms.EDProducer("PATPhotonProducer",
    # input collection
    photonSource = cms.InputTag("gedPhotons"),
    electronSource = cms.InputTag("gedGsfElectrons"),             
    beamLineSrc = cms.InputTag("offlineBeamSpot"),

    reducedBarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),             
             
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
    embedSeedCluster               = cms.bool(True),  ## embed in AOD externally stored the photon's seedcluster 
    embedBasicClusters             = cms.bool(True),  ## embed in AOD externally stored the photon's basic clusters 
    embedPreshowerClusters         = cms.bool(True),  ## embed in AOD externally stored the photon's preshower clusters 
    embedRecHits         = cms.bool(True),  ## embed in AOD externally stored the RecHits - can be called from the PATPhotonProducer 
    
    # embed IsoDeposits to recompute isolation
    isoDeposits = cms.PSet(),

    # user defined isolation variables the variables defined here will be accessible
    # via pat::Photon::userIsolation(IsolationKeys key) with the key as defined in
    # DataFormats/PatCandidates/interface/Isolation.h
    userIsolation = cms.PSet(
        #PFClusterEcalIso = cms.InputTag('electronEcalPFClusterIsolationProducer'),
        #PFClusterHcalIso = cms.InputTag('electronHcalPFClusterIsolationProducer'),
        ),

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
    resolutions     = cms.PSet(),

    # PFClusterIso
    addPFClusterIso = cms.bool(False)
)
