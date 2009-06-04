import FWCore.ParameterSet.Config as cms

allLayer1Electrons = cms.EDProducer("PATElectronProducer",
    # General configurables
    electronSource = cms.InputTag("electronsNoDuplicates"),

                                    
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

    # Embedding of AOD items
    embedTrack        = cms.bool(False), ## whether to embed in AOD externally stored track (note: gsf electrons don't have a track)
    embedGsfTrack     = cms.bool(True), ## whether to embed in AOD externally stored gsf track
    embedSuperCluster = cms.bool(True), ## whether to embed in AOD externally stored supercluster

    # resolution configurables
    addResolutions   = cms.bool(False),

    # pflow specific
    pfElectronSource = cms.InputTag("pfElectrons"),
    useParticleFlow =  cms.bool( False ),
    embedPFCandidate = cms.bool(False),

    # Store isolation values
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
    # Store IsoDeposits
    isoDeposits = cms.PSet(
        tracker = cms.InputTag("eleIsoDepositTk"),
        ecal    = cms.InputTag("eleIsoDepositEcalFromHits"),
        hcal    = cms.InputTag("eleIsoDepositHcalFromTowers"),
    ),


    # electron ID configurables
    addElectronID = cms.bool(True),
    electronIDSources = cms.PSet(
        # configure many IDs as InputTag <someName> = <someTag>
        # you can comment out those you don't want to save some disk space
        eidRobustLoose      = cms.InputTag("eidRobustLoose"),
        eidRobustTight      = cms.InputTag("eidRobustTight"),
        eidLoose            = cms.InputTag("eidLoose"),
        eidTight            = cms.InputTag("eidTight"),
        eidRobustHighEnergy = cms.InputTag("eidRobustHighEnergy"),
    ),

    # Trigger matching configurables
    addTrigMatch = cms.bool(True),
    # trigger primitive sources to be used for the matching
    trigPrimMatch = cms.VInputTag(
            cms.InputTag("electronTrigMatchHLT1ElectronRelaxed"), 
            cms.InputTag("electronTrigMatchCandHLT1ElectronStartup")
    ),

    # MC matching configurables
    addGenMatch      = cms.bool(True),
    embedGenMatch    = cms.bool(False),
    genParticleMatch = cms.InputTag("electronMatch"), ## Association between electrons and generator particles

    # Efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),
    
    # electron cluster shape configurables
    addElectronShapes = cms.bool(True),
    reducedBarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),

)


