import FWCore.ParameterSet.Config as cms
import PhysicsTools.PatAlgos.PATElectronProducer_cfi as _mod

patElectrons = _mod.PATElectronProducer.clone(
    # input collection
    electronSource = "gedGsfElectrons",

    # use particle flow instead of std reco
    pfElectronSource = "particleFlow",
    pfCandidateMap   = "particleFlow:electrons",

    # collections for mva input variables
    addMVAVariables = True,
    reducedBarrelRecHitCollection = "reducedEcalRecHitsEB",
    reducedEndcapRecHitCollection = "reducedEcalRecHitsEE",

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
    embedGsfElectronCore    = True,  ## embed in AOD externally stored gsf electron core
    embedGsfTrack           = False, ## embed in AOD externally stored gsf track
    embedSuperCluster       = True,  ## embed in AOD externally stored supercluster
    embedPflowSuperCluster  = True,  ## embed in AOD externally stored supercluster
    embedSeedCluster        = True,  ## embed in AOD externally stored the electron's seedcluster 
    embedBasicClusters      = True,  ## embed in AOD externally stored the electron's basic clusters 
    embedPreshowerClusters  = True,  ## embed in AOD externally stored the electron's preshower clusters 
    embedPflowBasicClusters = True,  ## embed in AOD externally stored the electron's pflow basic clusters 
    embedPflowPreshowerClusters = True,  ## embed in AOD externally stored the electron's pflow preshower clusters 
    embedPFCandidate        = True, ## embed in AOD externally stored particle flow candidate
    embedTrack              = True, ## embed in AOD externally stored track (note: gsf electrons don't have a track)
    embedRecHits            = True, ## embed in AOD externally stored the RecHits - can be called from the PATElectronProducer 

    # embed IsoDeposits to recompute isolation
    isoDeposits = cms.PSet(),

    # user defined isolation variables the variables defined here will be accessible
    # via pat::Electron::userIsolation(IsolationKeys key) with the key as defined in
    # DataFormats/PatCandidates/interface/Isolation.h
    userIsolation = dict(),

    # electron ID
    addElectronID = False,
    electronIDSources = cms.PSet(),

    # mc matching
    addGenMatch      = True,
    embedGenMatch    = True,
    genParticleMatch = "electronMatch", ## Association between electrons and generator particles

    # efficiencies
    addEfficiencies = False,
    efficiencies    = dict(),

    # resolution configurables
    addResolutions   = False,
    resolutions      = dict(),

    # high level selections
    embedHighLevelSelection = True,
    beamLineSrc             = "offlineBeamSpot",
    pvSrc                   = "offlinePrimaryVertices",

    # PFClusterIso
    addPFClusterIso     = False,
    ecalPFClusterIsoMap = "",
    hcalPFClusterIsoMap = "",
    addPuppiIsolation   = False,

    # Compute and store Mini-Isolation.
    # Implemention and a description of parameters can be found in:
    # PhysicsTools/PatUtils/src/PFIsolation.cc
    # only works in miniaod, so set to True in miniAOD_tools.py
    computeMiniIso    = False,
    pfCandsForMiniIso = "packedPFCandidates",
     # veto on candidates in deadcone only in endcap
    miniIsoParamsE = [0.05, 0.2, 10.0, 0.0, 0.015, 0.015, 0.08, 0.0, 0.0],
    miniIsoParamsB = [0.05, 0.2, 10.0, 0.0, 0.000, 0.000, 0.00, 0.0, 0.0],

)
del patElectrons.electronIDSource
