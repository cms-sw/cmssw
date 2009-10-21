import FWCore.ParameterSet.Config as cms

allLayer1Taus = cms.EDProducer("PATTauProducer",
    # input
    tauSource = cms.InputTag("fixedConePFTauProducer"),

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
      # add candidate ptrs here
      userCands = cms.PSet(
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
                           
    # embed IsoDeposits
    isoDeposits = cms.PSet(
        pfAllParticles = cms.InputTag("tauIsoDepositPFCandidates"),
        pfChargedHadron = cms.InputTag("tauIsoDepositPFChargedHadrons"),
        pfNeutralHadron = cms.InputTag("tauIsoDepositPFNeutralHadrons"),
        pfGamma = cms.InputTag("tauIsoDepositPFGammas")
    ),

    # user defined isolation variables the variables defined here will be accessible
    # via pat::Tau::userIsolation(IsolationKeys key) with the key as defined in
    # DataFormats/PatCandidates/interface/Isolation.h
    #
    # (set Pt thresholds for PFChargedHadrons (PFGammas) to 1.0 (1.5) GeV,
    # matching the thresholds used when computing the tau iso. discriminators
    # in RecoTauTag/RecoTau/python/PFRecoTauDiscriminationByIsolation_cfi.py)        
    userIsolation = cms.PSet(
        pfAllParticles = cms.PSet(
            src = cms.InputTag("tauIsoDepositPFCandidates"),
            deltaR = cms.double(0.5),
            threshold = cms.double(0.)
        ),
        pfChargedHadron = cms.PSet(
            src = cms.InputTag("tauIsoDepositPFChargedHadrons"),
            deltaR = cms.double(0.5),
            threshold = cms.double(0.)
        ),
        pfNeutralHadron = cms.PSet(
            src = cms.InputTag("tauIsoDepositPFNeutralHadrons"),
            deltaR = cms.double(0.5),
            threshold = cms.double(0.)
        ),
        pfGamma = cms.PSet(
            src = cms.InputTag("tauIsoDepositPFGammas"),
            deltaR = cms.double(0.5),
            threshold = cms.double(0.)
        )
    ),

    # tau ID (for efficiency studies)
    addTauID     = cms.bool(True),
    tauIDSources = cms.PSet(
        # configure many IDs as InputTag <someName> = <someTag>
        # you can comment out those you don't want to save some
        # disk space
        leadingTrackFinding = cms.InputTag("fixedConePFTauDiscriminationByLeadingTrackFinding"),
        leadingTrackPtCut = cms.InputTag("fixedConePFTauDiscriminationByLeadingTrackPtCut"),
        trackIsolation = cms.InputTag("fixedConePFTauDiscriminationByTrackIsolation"),
        ecalIsolation = cms.InputTag("fixedConePFTauDiscriminationByECALIsolation"),
        byIsolation = cms.InputTag("fixedConePFTauDiscriminationByIsolation"),
        againstElectron = cms.InputTag("fixedConePFTauDiscriminationAgainstElectron"),
        againstMuon = cms.InputTag("fixedConePFTauDiscriminationAgainstMuon")
    ),

    # tau decay mode configurables
    addDecayMode = cms.bool(False),
    decayModeSrc = cms.InputTag("fixedConePFTauDecayModeProducer"),                    

    # mc matching configurables
    addGenMatch      = cms.bool(True),
    embedGenMatch    = cms.bool(True),
    genParticleMatch = cms.InputTag("tauMatch"),
    addGenJetMatch   = cms.bool(True),
    embedGenJetMatch = cms.bool(True),    
    genJetMatch      = cms.InputTag("tauGenJetMatch"),

    # efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),

    # resolution
    addResolutions  = cms.bool(False),
    resolutions     = cms.PSet()
)


