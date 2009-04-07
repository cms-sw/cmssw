import FWCore.ParameterSet.Config as cms

allLayer1Taus = cms.EDProducer("PATTauProducer",
    # input
    tauSource = cms.InputTag("pfRecoTauProducer"),

    # add user data
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

    # embedding objects
    embedLeadTrack       = cms.bool(False), ## embed in AOD externally stored leading track
    embedSignalTracks    = cms.bool(False), ## embed in AOD externally stored signal tracks
    embedIsolationTracks = cms.bool(False), ## embed in AOD externally stored isolation tracks

    # isolation
    isolation = cms.PSet(
        pfAllParticles = cms.PSet(
            src = cms.InputTag("patLayer0PFTauIsolation", "tauIsoDepositPFCandidates"),
            deltaR = cms.double(0.5)
        ),
        pfChargedHadron = cms.PSet(
            src = cms.InputTag("patLayer0PFTauIsolation", "tauIsoDepositPFChargedHadrons"),
            deltaR = cms.double(0.5)
        ),
        pfNeutralHadron = cms.PSet(
            src = cms.InputTag("patLayer0PFTauIsolation", "tauIsoDepositPFNeutralHadrons"),
            deltaR = cms.double(0.5)
        ),
        pfGamma = cms.PSet(
            src = cms.InputTag("patLayer0PFTauIsolation", "tauIsoDepositPFGammas"),
            deltaR = cms.double(0.5)
        )
    ),                           
    # embed IsoDeposits
    isoDeposits = cms.PSet(
        pfAllParticles = cms.InputTag("patLayer0PFTauIsolation", "tauIsoDepositPFCandidates"),
        pfChargedHadron = cms.InputTag("patLayer0PFTauIsolation", "tauIsoDepositPFChargedHadrons"),
        pfNeutralHadron = cms.InputTag("patLayer0PFTauIsolation", "tauIsoDepositPFNeutralHadrons"),
        pfGamma = cms.InputTag("patLayer0PFTauIsolation", "tauIsoDepositPFGammas")
    ),

    # tau ID (for efficiency studies)
    addTauID     = cms.bool(True),
    tauIDSources = cms.PSet(
        # configure many IDs as InputTag <someName> = <someTag>
        # you can comment out those you don't want to save some
        # disk space
        leadingTrackFinding = cms.InputTag("pfRecoTauDiscriminationByLeadingTrackFinding"),
        leadingTrackPtCut = cms.InputTag("pfRecoTauDiscriminationByLeadingTrackPtCut"),
        trackIsolation = cms.InputTag("pfRecoTauDiscriminationByTrackIsolation"),
        ecalIsolation = cms.InputTag("pfRecoTauDiscriminationByECALIsolation"),
        byIsolation = cms.InputTag("pfRecoTauDiscriminationByIsolation"),
        againstElectron = cms.InputTag("pfRecoTauDiscriminationAgainstElectron"),
        againstMuon = cms.InputTag("pfRecoTauDiscriminationAgainstMuon")
    ),

    # trigger matching configurables
    addTrigMatch  = cms.bool(False),
    trigPrimMatch = cms.VInputTag(''),

    # mc matching configurables
    addGenMatch      = cms.bool(True),
    embedGenMatch    = cms.bool(False),
    genParticleMatch = cms.InputTag("tauMatch"),
    addGenJetMatch   = cms.bool(True),
    embedGenJetMatch = cms.bool(False),    
    genJetMatch      = cms.InputTag("tauGenJetMatch"),

    # efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),

    # resolution
    addResolutions  = cms.bool(False)
)


