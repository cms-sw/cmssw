import FWCore.ParameterSet.Config as cms

allLayer1Taus = cms.EDProducer("PATTauProducer",
    # General configurables
    tauSource = cms.InputTag("allLayer0Taus"),

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

    embedLeadTrack       = cms.bool(False), ## whether to embed in AOD externally stored leading track
    embedSignalTracks    = cms.bool(False), ## whether to embed in AOD externally stored signal tracks
    embedIsolationTracks = cms.bool(False), ## whether to embed in AOD externally stored isolation tracks

    # resolution configurables
    addResolutions = cms.bool(True),
    tauResoFile = cms.string('PhysicsTools/PatUtils/data/Resolutions_tau.root'),
    useNNResolutions = cms.bool(True), ## use the neural network approach?

    # isolation configurables
    # Store isolation values
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
    # Store IsoDeposits
    isoDeposits = cms.PSet(
        pfAllParticles = cms.InputTag("patLayer0PFTauIsolation", "tauIsoDepositPFCandidates"),
        pfChargedHadron = cms.InputTag("patLayer0PFTauIsolation", "tauIsoDepositPFChargedHadrons"),
        pfNeutralHadron = cms.InputTag("patLayer0PFTauIsolation", "tauIsoDepositPFNeutralHadrons"),
        pfGamma = cms.InputTag("patLayer0PFTauIsolation", "tauIsoDepositPFGammas")
    ),

    # tau ID configurables
    # (for efficiency studies)
    addTauID = cms.bool(True),
    tauIDSources = cms.PSet(
        # configure many IDs as InputTag <someName> = <someTag>
        # you can comment out those you don't want to save some disk space
        leadingTrackFinding = cms.InputTag("patPFRecoTauDiscriminationByLeadingTrackFinding"),
        leadingTrackPtCut = cms.InputTag("patPFRecoTauDiscriminationByLeadingTrackPtCut"),
        trackIsolation = cms.InputTag("patPFRecoTauDiscriminationByTrackIsolation"),
        ecalIsolation = cms.InputTag("patPFRecoTauDiscriminationByECALIsolation"),
        byIsolation = cms.InputTag("patPFRecoTauDiscriminationByIsolation"),
        againstElectron = cms.InputTag("patPFRecoTauDiscriminationAgainstElectron"),
        againstMuon = cms.InputTag("patPFRecoTauDiscriminationAgainstMuon")
    ),

    # Trigger matching configurables
    addTrigMatch = cms.bool(True),
    trigPrimMatch = cms.VInputTag(cms.InputTag("tauTrigMatchHLT1Tau")),

    # MC matching configurables
    addGenMatch = cms.bool(True),
    embedGenMatch = cms.bool(False),
    genParticleMatch = cms.InputTag("tauMatch"), ## particles source to be used for the matching

    # MC jet matching configurables
    addGenJetMatch = cms.bool(True),
    # the following is not used. ?
    embedGenJetMatch = cms.bool(False),
    
    genJetMatch = cms.InputTag("tauGenJetMatch"), ## particles source to be used for the matching

    # Efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet()
)


