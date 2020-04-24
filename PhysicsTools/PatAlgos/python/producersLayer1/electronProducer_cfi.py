import FWCore.ParameterSet.Config as cms

patElectrons = cms.EDProducer("PATElectronProducer",
    # input collection
    electronSource = cms.InputTag("gedGsfElectrons"),

    # use particle flow instead of std reco
    useParticleFlow  =  cms.bool( False ),
    pfElectronSource = cms.InputTag("particleFlow"),
    pfCandidateMap = cms.InputTag("particleFlow:electrons"),
    usePfCandidateMultiMap = cms.bool( False ),

    # collections for mva input variables
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
    embedGsfElectronCore = cms.bool(True),  ## embed in AOD externally stored gsf electron core
    embedGsfTrack        = cms.bool(False),  ## embed in AOD externally stored gsf track
    embedSuperCluster    = cms.bool(True),  ## embed in AOD externally stored supercluster
    embedPflowSuperCluster         = cms.bool(True),  ## embed in AOD externally stored supercluster
    embedSeedCluster               = cms.bool(True),  ## embed in AOD externally stored the electron's seedcluster 
    embedBasicClusters             = cms.bool(True),  ## embed in AOD externally stored the electron's basic clusters 
    embedPreshowerClusters         = cms.bool(True),  ## embed in AOD externally stored the electron's preshower clusters 
    embedPflowBasicClusters        = cms.bool(True),  ## embed in AOD externally stored the electron's pflow basic clusters 
    embedPflowPreshowerClusters    = cms.bool(True),  ## embed in AOD externally stored the electron's pflow preshower clusters 
    embedPFCandidate     = cms.bool(True),  ## embed in AOD externally stored particle flow candidate
    embedTrack           = cms.bool(True), ## embed in AOD externally stored track (note: gsf electrons don't have a track)
    embedRecHits         = cms.bool(True),  ## embed in AOD externally stored the RecHits - can be called from the PATElectronProducer 

    # embed IsoDeposits to recompute isolation
    isoDeposits = cms.PSet(),

    # user defined isolation variables the variables defined here will be accessible
    # via pat::Electron::userIsolation(IsolationKeys key) with the key as defined in
    # DataFormats/PatCandidates/interface/Isolation.h
    userIsolation = cms.PSet(),

    # electron ID
    addElectronID = cms.bool(True),
    electronIDSources = cms.PSet(
        # configure many IDs as InputTag <someName> = <someTag> you
        # can comment out those you don't want to save some disk space
        eidRobustLoose      = cms.InputTag("eidRobustLoose"),
        eidRobustTight      = cms.InputTag("eidRobustTight"),
        eidLoose            = cms.InputTag("eidLoose"),
        eidTight            = cms.InputTag("eidTight"),
        eidRobustHighEnergy = cms.InputTag("eidRobustHighEnergy"),
    ),

    # mc matching
    addGenMatch      = cms.bool(True),
    embedGenMatch    = cms.bool(True),
    genParticleMatch = cms.InputTag("electronMatch"), ## Association between electrons and generator particles

    # efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),

    # resolution configurables
    addResolutions   = cms.bool(False),
    resolutions      = cms.PSet(),

    # high level selections
    embedHighLevelSelection = cms.bool(True),
    beamLineSrc             = cms.InputTag("offlineBeamSpot"),
    pvSrc                   = cms.InputTag("offlinePrimaryVertices"),

    # PFClusterIso
    addPFClusterIso = cms.bool(False),
    addPuppiIsolation = cms.bool(False),

    # Compute and store Mini-Isolation.
    # Implemention and a description of parameters can be found in:
    # PhysicsTools/PatUtils/src/PFIsolation.cc
    # only works in miniaod, so set to True in miniAOD_tools.py
    computeMiniIso = cms.bool(False),
    pfCandsForMiniIso = cms.InputTag("packedPFCandidates"),
     # veto on candidates in deadcone only in endcap
    miniIsoParamsE = cms.vdouble(0.05, 0.2, 10.0, 0.0, 0.015, 0.015, 0.08, 0.0, 0.0),
    miniIsoParamsB = cms.vdouble(0.05, 0.2, 10.0, 0.0, 0.000, 0.000, 0.00, 0.0, 0.0),

)
