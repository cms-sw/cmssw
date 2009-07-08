import FWCore.ParameterSet.Config as cms

allLayer1Electrons = cms.EDProducer("PATElectronProducer",
    # input collection
    electronSource = cms.InputTag("electronsNoDuplicates"),

    # use particle flow instead of std reco    
    useParticleFlow  =  cms.bool( False ),
    pfElectronSource = cms.InputTag("pfElectrons"),
                                    
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
    embedTrack        = cms.bool(False), ## embed in AOD externally stored track (note: gsf electrons don't have a track)
    embedGsfTrack     = cms.bool(True),  ## embed in AOD externally stored gsf track
    embedSuperCluster = cms.bool(True),  ## embed in AOD externally stored supercluster
    embedPFCandidate  = cms.bool(False), ## embed in AOD externally stored particle flow candidate
                                    
    # isolation
    isolation = cms.PSet(
        tracker = cms.PSet(
            src = cms.InputTag("eleIsoFromDepsTk"),
        ),
        ecal = cms.PSet(
            src = cms.InputTag("eleIsoFromDepsEcalFromHits"),
        ),
        hcal = cms.PSet(
            src = cms.InputTag("eleIsoFromDepsHcalFromTowers"),
        ),
        user = cms.VPSet(),
    ),
    # embed IsoDeposits to recompute isolation
    isoDeposits = cms.PSet(
        tracker = cms.InputTag("eleIsoDepositTk"),
        ecal    = cms.InputTag("eleIsoDepositEcalFromHits"),
        hcal    = cms.InputTag("eleIsoDepositHcalFromTowers"),
    ),

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
    addGenMatch      = cms.bool(False),
    embedGenMatch    = cms.bool(False),
    genParticleMatch = cms.InputTag("electronMatch"), ## Association between electrons and generator particles
    
    # efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),

    # resolution configurables
    addResolutions   = cms.bool(False),
    resolutions      = cms.PSet(),
)


