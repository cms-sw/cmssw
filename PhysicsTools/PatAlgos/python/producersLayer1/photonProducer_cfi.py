import FWCore.ParameterSet.Config as cms
import PhysicsTools.PatAlgos.PATPhotonProducer_cfi as _mod

patPhotons = _mod.PATPhotonProducer.clone(
    # input collection
    photonSource   = "gedPhotons",
    electronSource = "gedGsfElectrons",
    beamLineSrc    = "offlineBeamSpot",

    # user data to add
    userData = dict(
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
      userFunctions = [],
      userFunctionLabels = []
    ),

    # embedding of AOD items
    embedSuperCluster      = True, ## whether to embed in AOD externally stored supercluster
    embedSeedCluster       = True, ## embed in AOD externally stored the photon's seedcluster 
    embedBasicClusters     = True, ## embed in AOD externally stored the photon's basic clusters 
    embedPreshowerClusters = True, ## embed in AOD externally stored the photon's preshower clusters 
    embedRecHits           = True, ## embed in AOD externally stored the RecHits - can be called from the PATPhotonProducer 
    saveRegressionData     = True, ## save regression input variables
    
    # embed IsoDeposits to recompute isolation
    isoDeposits = cms.PSet(),

    # user defined isolation variables the variables defined here will be accessible
    # via pat::Photon::userIsolation(IsolationKeys key) with the key as defined in
    # DataFormats/PatCandidates/interface/Isolation.h
    userIsolation = dict(
        #PFClusterEcalIso = 'electronEcalPFClusterIsolationProducer',
        #PFClusterHcalIso = 'electronHcalPFClusterIsolationProducer',
        ),

    # photon ID
    addPhotonID     = False,
    photonIDSources = cms.PSet(),
    # mc matching
    addGenMatch      = True,
    embedGenMatch    = True,
    genParticleMatch = "photonMatch", ## particles source to be used for the matching

    # efficiencies
    addEfficiencies = False,
    efficiencies    = dict(),

    # resolutions
    addResolutions  = False,
    resolutions     = dict(),

    # PFClusterIso
    addPFClusterIso     = False,
    ecalPFClusterIsoMap = "",
    hcalPFClusterIsoMap = "",
    addPuppiIsolation   = False
)
del patPhotons.photonIDSource
