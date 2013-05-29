import FWCore.ParameterSet.Config as cms

patTaus = cms.EDProducer("PATTauProducer",
    # input
    tauSource = cms.InputTag("shrinkingConePFTauProducer"),

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

    # jet energy corrections
    addTauJetCorrFactors = cms.bool(False),
    tauJetCorrFactorsSource = cms.VInputTag(cms.InputTag("patTauJetCorrFactors")),                     

    # embedding objects (for Calo- and PFTaus)
    embedLeadTrack = cms.bool(False), ## embed in AOD externally stored leading track
    embedSignalTracks = cms.bool(False), ## embed in AOD externally stored signal tracks
    embedIsolationTracks = cms.bool(False), ## embed in AOD externally stored isolation tracks
    # embedding objects (for PFTaus only)
    embedLeadPFCand = cms.bool(False), ## embed in AOD externally stored leading PFCandidate
    embedLeadPFChargedHadrCand = cms.bool(False), ## embed in AOD externally stored leading PFChargedHadron candidate
    embedLeadPFNeutralCand = cms.bool(False), ## embed in AOD externally stored leading PFNeutral Candidate
    embedSignalPFCands = cms.bool(False), ## embed in AOD externally stored signal PFCandidates
    embedSignalPFChargedHadrCands = cms.bool(False), ## embed in AOD externally stored signal PFChargedHadronCandidates
    embedSignalPFNeutralHadrCands = cms.bool(False), ## embed in AOD externally stored signal PFNeutralHadronCandidates
    embedSignalPFGammaCands = cms.bool(False), ## embed in AOD externally stored signal PFGammaCandidates
    embedIsolationPFCands = cms.bool(False), ## embed in AOD externally stored isolation PFCandidates
    embedIsolationPFChargedHadrCands = cms.bool(False), ## embed in AOD externally stored isolation PFChargedHadronCandidates
    embedIsolationPFNeutralHadrCands = cms.bool(False), ## embed in AOD externally stored isolation PFNeutralHadronCandidates
    embedIsolationPFGammaCands = cms.bool(False), ## embed in AOD externally stored isolation PFGammaCandidates

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
        leadingTrackFinding = cms.InputTag("shrinkingConePFTauDiscriminationByLeadingTrackFinding"),
        #leadingTrackPtCut = cms.InputTag("shrinkingConePFTauDiscriminationByLeadingTrackPtCut"),
        leadingPionPtCut = cms.InputTag("shrinkingConePFTauDiscriminationByLeadingPionPtCut"),
        #trackIsolation = cms.InputTag("shrinkingConePFTauDiscriminationByTrackIsolation"),
        #trackIsolationUsingLeadingPion = cms.InputTag("shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion"),
        #ecalIsolation = cms.InputTag("shrinkingConePFTauDiscriminationByECALIsolation"),
        #ecalIsolationUsingLeadingPion = cms.InputTag("shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion"),
        #byIsolation = cms.InputTag("shrinkingConePFTauDiscriminationByIsolation"),
        byIsolationUsingLeadingPion = cms.InputTag("shrinkingConePFTauDiscriminationByIsolationUsingLeadingPion"),
        againstElectron = cms.InputTag("shrinkingConePFTauDiscriminationAgainstElectron"),
        againstMuon = cms.InputTag("shrinkingConePFTauDiscriminationAgainstMuon"),
        #byTaNC = cms.InputTag("shrinkingConePFTauDiscriminationByTaNC"),
        #byTaNCfrOnePercent = cms.InputTag("shrinkingConePFTauDiscriminationByTaNCfrOnePercent"),
        #byTaNCfrHalfPercent = cms.InputTag("shrinkingConePFTauDiscriminationByTaNCfrHalfPercent"),
        #byTaNCfrQuarterPercent = cms.InputTag("shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent"),
        #byTaNCfrTenthPercent = cms.InputTag("shrinkingConePFTauDiscriminationByTaNCfrTenthPercent")
    ),

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


