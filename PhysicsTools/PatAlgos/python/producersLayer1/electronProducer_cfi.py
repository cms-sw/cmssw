import FWCore.ParameterSet.Config as cms

patElectrons = cms.EDProducer("PATElectronProducer",
    # input collection
    electronSource = cms.InputTag("gsfElectrons"),

    # use particle flow instead of std reco    
    useParticleFlow  =  cms.bool( False ),
    pfElectronSource = cms.InputTag("pfIsolatedElectrons"),
    linkToPFSource   = cms.InputTag(""),
                                    
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
    embedGsfTrack        = cms.bool(True),  ## embed in AOD externally stored gsf track
    embedSuperCluster    = cms.bool(True),  ## embed in AOD externally stored supercluster
    embedPFCandidate     = cms.bool(True),  ## embed in AOD externally stored particle flow candidate
    embedTrack           = cms.bool(False), ## embed in AOD externally stored track (note: gsf electrons don't have a track)
                                    
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
    usePV                   = cms.bool(True),                          
    beamLineSrc             = cms.InputTag("offlineBeamSpot"),
    pvSrc                   = cms.InputTag("offlinePrimaryVertices")
)
